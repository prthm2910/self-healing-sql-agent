from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workflow.state import State
from src.utils.logger import logger
from src.services.sql_engine import sql_engine
from src.workflow.schema.discovery import ClassifierOutput, AnchorSelection, SchemaSelectorOutput
from src.workflow.nodes.base import BaseNode, llm
from src.prompts.discovery import (
    get_classifier_prompt,
    get_entity_extraction_prompt,
    get_physical_mapping_prompt,
    get_schema_pruning_prompt
)


class ClassifierNode(BaseNode):
    """Determines if the SQL query is SIMPLE (one table) or COMPLEX (joins/logic)."""
    name = "classifier_node"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        # Get last 3 human messages for context (handles "try again" or follow-ups)
        human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-3:]
        context_msg = " | ".join(human_msgs)
        last_msg = human_msgs[-1] if human_msgs else ""
        
        logger.info(f"Node: classifier_node | Classifying message: {last_msg[:30]} | Context: {context_msg[:50]}...")
        
        prompt_template = get_classifier_prompt()
        prompt_val = prompt_template.invoke({
            "context_msg": context_msg,
            "last_msg": last_msg
        })
        chain = llm.with_structured_output(ClassifierOutput)
        res = self.robust_invoke(chain, prompt_val.to_messages(), ClassifierOutput)
        res.node_name = "classifier"
        
        logger.info(f"Classification: {'COMPLEX' if res.is_complex else 'SIMPLE'} | Thought: {res.thought_process}")
        
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())

        return {"is_complex": res.is_complex, "agent_logs": logs}


class AnchorSelectorNode(BaseNode):
    """Hybrid Discovery Phase 1: Two-Pass Entity Extraction & Physical Table Mapping."""
    name = "anchor_selector"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        last_msg = state["messages"][-1].content
        all_tables = sql_engine.list_tables()

        # Pass 1: Semantic Entity Extraction (Structured)
        entity_template = get_entity_extraction_prompt()
        entity_prompt_val = entity_template.invoke({"last_msg": last_msg})
        entity_chain = llm.with_structured_output(AnchorSelection)
        entity_res = self.robust_invoke(entity_chain, entity_prompt_val.to_messages(), AnchorSelection)
        entities = ", ".join(entity_res.anchors)

        # Pass 2: Hard Physical Table Mapping
        mapping_template = get_physical_mapping_prompt()
        mapping_prompt_val = mapping_template.invoke({
            "entities": entities,
            "all_tables": all_tables
        })
        anchor_chain = llm.with_structured_output(AnchorSelection)
        anchor_res = self.robust_invoke(anchor_chain, mapping_prompt_val.to_messages(), AnchorSelection)
        anchors = [a for a in anchor_res.anchors if a in all_tables]

        # Deterministic Physical Table Mapping Fallback (Post-processing)
        last_msg_lower = last_msg.lower()
        
        # Payment keywords
        payment_keywords = ["spent", "amount", "revenue", "payment", "paid", "sales", "price", "income"]
        if any(kw in last_msg_lower for kw in payment_keywords) and "payment" in all_tables:
            if "payment" not in anchors:
                logger.info("Deterministic injection: Adding 'payment' table based on query keywords.")
                anchors.append("payment")
                
        # Category keywords
        category_keywords = ["category", "genre", "categories", "action", "animation", "children", "classics", "comedy", "documentary", "drama", "family", "foreign", "games", "horror", "music", "new", "sci-fi", "sports", "travel"]
        if any(kw in last_msg_lower for kw in category_keywords) and "category" in all_tables:
            if "category" not in anchors:
                logger.info("Deterministic injection: Adding 'category' table based on query keywords.")
                anchors.append("category")
                
        # Country keywords
        country_keywords = ["country", "countries", "canada", "geographic", "address", "city"]
        if any(kw in last_msg_lower for kw in country_keywords) and "country" in all_tables:
            if "country" not in anchors:
                logger.info("Deterministic injection: Adding 'country' table based on query keywords.")
                anchors.append("country")

        # 3. Deterministic FK Bridge Traversal
        bridges = sql_engine.get_bridge_tables(anchors)
        selected_tables = list(set(anchors + bridges))

        logger.info(f"Join Topology: Anchors={anchors} | Bridges={bridges}")

        logs = state.get("agent_logs", [])
        logs.append({
            "node_name": "anchor_selector",
            "anchors": anchors,
            "bridges": bridges,
            "selected_tables": selected_tables,
            "thought_process": getattr(anchor_res, "thought_process", "")
        })

        return {"selected_tables": selected_tables, "agent_logs": logs}


class ColumnPrunerNode(BaseNode):
    """Hybrid Discovery Phase 2: Surgically prunes columns while protecting Join Keys."""
    name = "column_pruner"

    def execute(self, state: State, config: RunnableConfig, user_id: str, thread_id: str, store=None, **kwargs) -> Dict[str, Any]:
        last_msg = state["messages"][-1].content
        selected_tables = state["selected_tables"]
        
        # Fetch relationships and partial schema
        fk_relationships = sql_engine.get_relevant_fks(selected_tables)
        partial_schema = sql_engine.get_schema(selected_tables)
        
        pruning_template = get_schema_pruning_prompt()
        pruning_prompt_val = pruning_template.invoke({
            "last_msg": last_msg,
            "partial_schema": partial_schema,
            "fk_relationships": fk_relationships
        })
        chain = llm.with_structured_output(SchemaSelectorOutput)
        res = self.robust_invoke(chain, pruning_prompt_val.to_messages(), SchemaSelectorOutput)
        res.node_name = "column_pruner"
        
        # Convert List[ColumnSelection] to Dict[str, List[str]] for state compatibility
        pruned_cols = {item.table_name: item.columns for item in res.selected_columns}
        # Convert List[FKRelationship] to List[Dict] for state compatibility
        pruned_fks = [rel.model_dump() for rel in res.fk_relationships]
        
        # Deterministic Guard: Ensure all selected tables exist and Join Keys are preserved
        for table in selected_tables:
            if table not in pruned_cols: pruned_cols[table] = []
            
        for rel in fk_relationships:
            if rel["source_column"] not in pruned_cols[rel["source_table"]]:
                pruned_cols[rel["source_table"]].append(rel["source_column"])
            if rel["target_column"] not in pruned_cols[rel["target_table"]]:
                pruned_cols[rel["target_table"]].append(rel["target_column"])
        
        logger.info(f"Pruning complete for {len(pruned_cols)} tables.")
        
        logs = state.get("agent_logs", [])
        logs.append(res.model_dump())
        
        return {
            "selected_columns": pruned_cols, 
            "fk_relationships": pruned_fks,
            "agent_logs": logs
        }


# Instantiate node callable objects
classifier_node = ClassifierNode()
anchor_selector_node = AnchorSelectorNode()
column_pruner_node = ColumnPrunerNode()
