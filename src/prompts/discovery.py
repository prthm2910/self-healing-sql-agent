# ### --- IMPORTS --- ###
from langchain_core.prompts import ChatPromptTemplate

# ##############################################################################
# [Elaborative Breakdown] Multi-Pass Schema Discovery & Intent Classification
# Why multi-pass schema discovery?
# When dealing with large relational schemas (e.g., Pagila's DVD rental database),
# dumping the entire raw schema (all tables, columns, and keys) into the LLM prompt 
# context creates substantial noise. This dilutes the LLM's attention, leads to 
# hallucinations (e.g., using non-existent table joins), and wastes tokens.
#
# Our multi-pass architecture splits this complex problem into smaller, highly specialized 
# sub-tasks, each backed by a distinct prompt:
#
# 1. Complexity Classification (`get_classifier_prompt`):
#    A quick, low-cost pass determining if a query requires multi-table joins (COMPLEX) or
#    runs against a single table (SIMPLE). This routes simple queries down an optimized 
#    one-step path and keeps complex logic isolated.
# 2. Semantic Entity Extraction (`get_entity_extraction_prompt`):
#    Extracts natural language database entity filters (e.g., "Canada", "Action") without 
#    focusing on physical database table structures yet.
# 3. Physical Table Mapping (`get_physical_mapping_prompt`):
#    Forces hard structural mapping of semantic entities to concrete physical tables
#    (e.g., mapping "Canada" to `country` table, "Action" to `category` table). It enforces 
#    strict rules such as avoiding read-only helper views.
# 4. Schema Pruning (`get_schema_pruning_prompt`):
#    Receives the mapped tables and trims the active schema to only the required columns 
#    and join keys, maintaining AST structural integrity while minimizing context.
# ##############################################################################


# ### --- CLASSIFIER PROMPT FACTORY --- ###

def get_classifier_prompt() -> ChatPromptTemplate:
    """
    Factory function for the SQL complexity intent classification prompt template.
    
    Determines if the user interaction requires a multi-table join or simple single-table operations.
    
    Returns:
        A compiled ChatPromptTemplate for complexity classification.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL Query Planner. 
Analyze the user interaction and determine if the current request requires joining multiple tables (COMPLEX) or if it is a single-table query (SIMPLE).

### CONTEXT (Last messages):
{context_msg}

### CURRENT QUESTION TO CLASSIFY:
"{last_msg}"

### EXAMPLES:
- "What are the first 10 films?" -> SIMPLE
- "Canada Action films" -> COMPLEX
- "try again" (where previous was Canada films) -> COMPLEX
- "count them" (where previous was customers) -> SIMPLE

### GUIDELINES (Output JSON keys: "is_complex", "thought_process"):
- If the current message is a follow-up like "try again", "run it", or "go ahead", use the PREVIOUS context to decide.
- If any geographic (Country/City) or category filters were mentioned recently, it is COMPLEX.
""")
    ])


# ### --- ENTITY EXTRACTION PROMPT FACTORY --- ###

def get_entity_extraction_prompt() -> ChatPromptTemplate:
    """
    Factory function for the Semantic Entity Extraction prompt template.
    
    Extracts semantic search keywords, locations, or filters to identify query anchor points.
    
    Returns:
        A compiled ChatPromptTemplate for entity extraction.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """Identify the core database entities and filters mentioned in this question.

Question: "{last_msg}"
Example Entities: 'Canada', 'Action', 'most rentals', 'spent'.

### INSTRUCTIONS:
You must call the AnchorSelection tool.
- Populate the 'anchors' field with the list of identified database entities or filters.
- Populate the 'thought_process' field with your detailed internal reasoning.
Both fields are strictly required. Do not leave 'thought_process' empty.
""")
    ])


# ### --- PHYSICAL MAPPING PROMPT FACTORY --- ###

def get_physical_mapping_prompt() -> ChatPromptTemplate:
    """
    Factory function for mapping semantic entities to physical database tables.
    
    Enforces strict architectural boundaries to block read-only view lookups and resolve
    ambiguous terms to concrete table joins.
    
    Returns:
        A compiled ChatPromptTemplate for physical database table mapping.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Database Architect. Map the following entities to the specific PHYSICAL tables needed to query them.

Entities Found: {entities}
Available Tables: {all_tables}

### CRITICAL RULES:
- If 'Canada' or geographic filters are mentioned, include 'country'.
- If 'Action' or categories are mentioned, include 'category'.
- If 'spent', 'amount', 'revenue', 'payment', 'paid', or 'sales' is mentioned, include 'payment'.
- NEVER select views (ending in '_info' or '_list').

### INSTRUCTIONS:
You must call the AnchorSelection tool.
- Populate the 'anchors' field with the specific physical table names (e.g. ['customer', 'rental']).
- Populate the 'thought_process' field with your step-by-step table mapping reasoning.
Both fields are strictly required. Do not leave 'thought_process' empty.
""")
    ])


# ### --- SCHEMA PRUNING PROMPT FACTORY --- ###

def get_schema_pruning_prompt() -> ChatPromptTemplate:
    """
    Factory function for pruning columns and foreign keys to create a minimal active schema query.
    
    Ensures join keys are preserved while omitting irrelevant column noise.
    
    Returns:
        A compiled ChatPromptTemplate for schema pruning.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Data Architect. Prune the schema below to ONLY include the columns needed for this question. Output MUST be valid JSON with keys: "selected_tables", "selected_columns", "fk_relationships", "fk_path_identified", "thought_process".
Question: "{last_msg}"
Schema:
{partial_schema}
Relationships:
{fk_relationships}

### CRITICAL RULES:
1. You MUST retain ALL columns mentioned in the 'Relationships' list (Join Keys).
2. Retain columns needed for filters (WHERE), ordering (ORDER BY), and display (SELECT).
""")
    ])

