# ### --- [IMPORTS] --- ###

from typing import Dict, Any, List, Optional

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


# ### --- [CLASSIFIER NODE] --- ###

class ClassifierNode(BaseNode):
    """
    Determines if the SQL query is SIMPLE (one table) or COMPLEX (joins/logic).
    
    This node serves as the early triage router in the workflow state machine.
    By parsing recent user input history, it evaluates whether the query requires
    a multi-table join and advanced task decomposition or can be addressed by a
    direct, single-table query.
    """
    name: str = "classifier_node"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Classifies user query complexity based on state messaging history.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: State changes updates merging into active state.
        """
        # 1. Context Extraction: Retrieve the last 3 user messages to capture conversational history.
        # This is critical for resolving context in multi-turn dialogues (e.g., resolving pronouns or "try again" requests).
        human_msgs: List[str] = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-3:]
        context_msg: str = " | ".join(human_msgs)
        last_msg: str = human_msgs[-1] if human_msgs else ""
        
        logger.info(f"Node: classifier_node | Classifying message: {last_msg[:30]} | Context: {context_msg[:50]}...")
        
        # 2. Invoke Classifier Prompt: Compile the user request and context into the prompt template.
        prompt_template = get_classifier_prompt()
        prompt_val = prompt_template.invoke({
            "context_msg": context_msg,
            "last_msg": last_msg
        })
        
        # 3. Call rate-limited LLM: Execute the structured output model using robust fallback parsing.
        chain = llm.with_structured_output(ClassifierOutput)
        res: ClassifierOutput = self.robust_invoke(chain, prompt_val.to_messages(), ClassifierOutput)
        res.node_name = "classifier"
        
        logger.info(f"Classification: {'COMPLEX' if res.is_complex else 'SIMPLE'} | Thought: {res.thought_process}")
        
        # 4. Save execution trace: Append the structured output log back into state for audit and debugging.
        logs: List[Dict[str, Any]] = state.get("agent_logs", [])
        logs.append(res.model_dump())

        # 5. State update: Return the decision and logs.
        return {"is_complex": res.is_complex, "agent_logs": logs}


# ### --- [ANCHOR SELECTOR NODE] --- ###

class AnchorSelectorNode(BaseNode):
    """
    Hybrid Discovery Phase 1: Two-Pass Entity Extraction & Physical Table Mapping.
    
    This node bridges the natural language entities present in the query to the
    exact, physical relational database tables. It performs semantic entity 
    extraction, maps those to candidate database tables, applies deterministic 
    domain-keyword fallbacks, and executes a Breadth-First Search (BFS) over foreign 
    key metadata to discover bridge tables needed to build a connected join topology.
    """
    name: str = "anchor_selector"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes structural entity extraction and maps physical relational tables.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: Identified selected_tables and tracking logs.
        """
        # 1. Schema Discovery Context: Get the latest query text and all physical user table names.
        last_msg: str = state["messages"][-1].content
        all_tables: List[str] = sql_engine.list_tables()

        # Pass 1: Semantic Entity Extraction (Structured LLM Call)
        # We prompt the LLM to extract domain-level entities (e.g., "movies", "categories", "clients").
        # Isolating semantic entities here avoids directly guessing database tables, reducing hallucination.
        entity_template = get_entity_extraction_prompt()
        entity_prompt_val = entity_template.invoke({"last_msg": last_msg})
        entity_chain = llm.with_structured_output(AnchorSelection)
        entity_res: AnchorSelection = self.robust_invoke(entity_chain, entity_prompt_val.to_messages(), AnchorSelection)
        entities: str = ", ".join(entity_res.anchors)

        # Pass 2: Hard Physical Table Mapping (Structured LLM Call)
        # Using the semantic entities discovered in Pass 1, we map them directly to the database's actual
        # physical tables. This keeps the LLM focused on exact matching without guessing.
        mapping_template = get_physical_mapping_prompt()
        mapping_prompt_val = mapping_template.invoke({
            "entities": entities,
            "all_tables": all_tables
        })
        anchor_chain = llm.with_structured_output(AnchorSelection)
        anchor_res: AnchorSelection = self.robust_invoke(anchor_chain, mapping_prompt_val.to_messages(), AnchorSelection)
        anchors: List[str] = [a for a in anchor_res.anchors if a in all_tables]

        # Deterministic Physical Table Mapping Fallback (Post-processing)
        # We use simple string-matching fallbacks to guarantee that key domain tables are injected
        # if specific semantic keywords are present in the raw query, mitigating LLM classification omissions.
        last_msg_lower: str = last_msg.lower()
        
        # Payment keywords mapping fallback
        payment_keywords: List[str] = ["spent", "amount", "revenue", "payment", "paid", "sales", "price", "income"]
        if any(kw in last_msg_lower for kw in payment_keywords) and "payment" in all_tables:
            if "payment" not in anchors:
                logger.info("Deterministic injection: Adding 'payment' table based on query keywords.")
                anchors.append("payment")
                
        # Category keywords mapping fallback
        category_keywords: List[str] = ["category", "genre", "categories", "action", "animation", "children", "classics", "comedy", "documentary", "drama", "family", "foreign", "games", "horror", "music", "new", "sci-fi", "sports", "travel"]
        if any(kw in last_msg_lower for kw in category_keywords) and "category" in all_tables:
            if "category" not in anchors:
                logger.info("Deterministic injection: Adding 'category' table based on query keywords.")
                anchors.append("category")
                
        # Country keywords mapping fallback
        country_keywords: List[str] = ["country", "countries", "canada", "geographic", "address", "city"]
        if any(kw in last_msg_lower for kw in country_keywords) and "country" in all_tables:
            if "country" not in anchors:
                logger.info("Deterministic injection: Adding 'country' table based on query keywords.")
                anchors.append("country")

        # 3. Deterministic FK Bridge Traversal
        # We treat the schema as a graph and use BFS search to discover intermediate bridge tables (e.g. film_actor, inventory)
        # required to physically connect anchor tables in a valid JOIN topology.
        bridges: List[str] = sql_engine.get_bridge_tables(anchors)
        selected_tables: List[str] = list(set(anchors + bridges))

        logger.info(f"Join Topology: Anchors={anchors} | Bridges={bridges}")

        # 4. Save Execution Trace
        logs: List[Dict[str, Any]] = state.get("agent_logs", [])
        logs.append({
            "node_name": "anchor_selector",
            "anchors": anchors,
            "bridges": bridges,
            "selected_tables": selected_tables,
            "thought_process": getattr(anchor_res, "thought_process", "")
        })

        return {"selected_tables": selected_tables, "agent_logs": logs}


# ### --- [COLUMN PRUNER NODE] --- ###

class ColumnPrunerNode(BaseNode):
    """
    Hybrid Discovery Phase 2: Surgically prunes columns while protecting Join Keys.
    
    Acts as a schema pruning optimizer to reduce total context window usage. 
    Using semantic selection, it determines which columns are relevant to the query 
    intent while deterministically injecting foreign key columns to ensure that 
    downstream joins remain mathematically viable.
    """
    name: str = "column_pruner"

    def execute(
        self, 
        state: State, 
        config: RunnableConfig, 
        user_id: str, 
        thread_id: str, 
        store: Optional[Any] = None, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes schema and foreign key pruning optimizations.

        Args:
            state (State): Active workflow state.
            config (RunnableConfig): Runnable execution config.
            user_id (str): Caller identifier.
            thread_id (str): Chat session identifier.
            store (Optional[Any]): Persistence store instance.
            **kwargs (Any): Dynamic node arguments.

        Returns:
            Dict[str, Any]: Pruned selected_columns, fk_relationships, and agent logs.
        """
        # 1. Fetch Discovery States: Retrieve recent question and previously selected active tables.
        last_msg: str = state["messages"][-1].content
        selected_tables: List[str] = state["selected_tables"]
        
        # 2. Extract Catalog Schema Metadata: Get active foreign key relationships and schema columns
        # strictly mapped to our active table selections, saving tokens by ignoring unrelated tables.
        fk_relationships: List[Dict[str, Any]] = sql_engine.get_relevant_fks(selected_tables)
        partial_schema: Dict[str, List[str]] = sql_engine.get_schema(selected_tables)
        
        # 3. Call LLM Schema Optimizer: Pruning schema context down to only semantic columns
        # required by user query intent while retaining join relations.
        pruning_template = get_schema_pruning_prompt()
        pruning_prompt_val = pruning_template.invoke({
            "last_msg": last_msg,
            "partial_schema": partial_schema,
            "fk_relationships": fk_relationships
        })
        chain = llm.with_structured_output(SchemaSelectorOutput)
        res: SchemaSelectorOutput = self.robust_invoke(chain, pruning_prompt_val.to_messages(), SchemaSelectorOutput)
        res.node_name = "column_pruner"
        
        # 4. Format outputs: Convert models to standard dict/list structures compatible with state validation.
        pruned_cols: Dict[str, List[str]] = {item.table_name: item.columns for item in res.selected_columns}
        pruned_fks: List[Dict[str, Any]] = [rel.model_dump() for rel in res.fk_relationships]
        
        # 5. Deterministic Join Column Protection:
        # Loop through active selected tables and physical foreign key keys.
        # Ensure that join keys are deterministically added back into pruned table column lists
        # even if the LLM semantically ignored them, protecting downstream relational joins from breaking.
        for table in selected_tables:
            if table not in pruned_cols: 
                pruned_cols[table] = []
            
        for rel in fk_relationships:
            if rel["source_column"] not in pruned_cols[rel["source_table"]]:
                pruned_cols[rel["source_table"]].append(rel["source_column"])
            if rel["target_column"] not in pruned_cols[rel["target_table"]]:
                pruned_cols[rel["target_table"]].append(rel["target_column"])
        
        logger.info(f"Pruning complete for {len(pruned_cols)} tables.")
        
        # 6. Save execution logs for observability.
        logs: List[Dict[str, Any]] = state.get("agent_logs", [])
        logs.append(res.model_dump())
        
        return {
            "selected_columns": pruned_cols, 
            "fk_relationships": pruned_fks,
            "agent_logs": logs
        }


# ### --- [NODE INSTANTIATION] --- ###

# Instantiate node callable objects
classifier_node = ClassifierNode()
anchor_selector_node = AnchorSelectorNode()
column_pruner_node = ColumnPrunerNode()


