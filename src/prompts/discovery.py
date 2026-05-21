from langchain_core.prompts import ChatPromptTemplate

def get_classifier_prompt():
    """
    Prompt for classifying the complexity of a user query.
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

def get_entity_extraction_prompt():
    """
    Pass 1: Semantic Entity Extraction prompt.
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

def get_physical_mapping_prompt():
    """
    Pass 2: Hard Physical Table Mapping prompt.
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

def get_schema_pruning_prompt():
    """
    Prunes the schema to only include relevant columns for the query.
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
