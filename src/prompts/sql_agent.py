from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_sql_generation_prompt():
    """
    Prompt factory for the initial SQL generation step with Lessons.
    Targets QueryBlueprint structured output.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert for the 'Pagila' DVD rental database.

### SYSTEMIC LESSONS (PAST MISTAKES)
{lessons}

DATABASE SCHEMA:
{schema}

### OBJECTIVE:
Generate a structured 'QueryBlueprint' JSON object to answer the user's question.

STRICT RULES:
1. NO HALLUCINATIONS: Do not assume columns exist (e.g., 'name', 'country') unless they are in the SCHEMA.
2. JOIN INTEGRITY: To filter by 'country', you MUST join via 'address' -> 'city' -> 'country'.
3. STRING MATCHING: Use 'ILIKE' for case-insensitive matches.
4. LIMIT: Default limit is 50.
5. Provide your 'thought_process' explaining your logic.
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

def get_sql_healing_prompt():
    """
    Prompt factory for the self-healing step when a query fails.
    Targets QueryBlueprint structured output.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL debugging expert. 
A previous SQL query failed. FIX it based on the error and schema by generating a new 'QueryBlueprint'.

DATABASE SCHEMA:
{schema}

FAILED QUERY: {failed_query}
ERROR MESSAGE: {error_message}

STRICT RULES:
1. Resolve the error using valid PostgreSQL logic.
2. Output MUST be a structured 'QueryBlueprint' JSON.
3. Provide your 'thought_process' explaining the fix.
"""),
        ("human", "Fix the query for: {question}")
    ])

def get_decomposer_prompt():
    """Manager prompt for planning complex queries."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL Architect. Decompose complex questions into atomic, joinable sub-tasks.
        
### RULES:
1. Sub-tasks must be atomic (focus on one logical part of the data).
2. 'task_id' must be 'task_1', 'task_2', etc.
3. 'required_columns' MUST include all keys needed for later joins.
4. 'join_plan' must explicitly connect ALL sub-tasks using the base task.
5. Return ONLY a valid JSON object matching the DecomposerOutput schema.

### SCHEMA CONTEXT:
{skeleton_schema}
"""),
        ("human", "Question: {question}")
    ])

def get_worker_prompt():
    """
    Prompt factory for the ReliableWorker node to solve atomic sub-tasks.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Reliable SQL Worker. Solve the following ATOMIC sub-task for the Pagila database.

REQUIRED JOIN KEYS: {required_columns} (These MUST be in your SELECT list)
SCHEMA:
{schema}

RULES:
- Return ONLY valid PostgreSQL.
- No semicolons.
- Use explicit aliases (e.g., 't.' for table name).
- Ensure ALL {required_columns} are in the SELECT list so they can be joined later.
"""),
        ("human", "{task_description}")
    ])

def get_sql_response_format_prompt():
    """
    Prompt factory for turning raw SQL data into a natural language summary.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a concise Data Analyst for the 'Pagila' DVD rental store.
Your goal is to provide a natural language 'summary' of the database results.

USER QUESTION: {question}
SQL EXECUTED: {query}
DATA SAMPLE: {data}

### STRICT GROUNDEDNESS RULES:
1. ONLY use the provided 'DATA SAMPLE' to answer the question.
2. If 'DATA SAMPLE' is empty or '[]', you MUST state that no results were found for the criteria.
3. NEVER invent names, counts, or statistics (e.g., actor names, rental counts) that are not present in the data.
4. If the SQL query appears to have failed or returned zero results, do NOT try to be "helpful" by using your general knowledge of movies.
5. Provide a 'summary' that directly answers the user's question (e.g., "The top 10 most expensive films are...").
6. Provide your 'thought_process' explaining how you derived the answer from the data.
"""),
        ("human", "Summarize these results.")
    ])
