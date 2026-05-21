# ### --- IMPORTS --- ###
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ##############################################################################
# [Elaborative Breakdown] Prompt Engineering for Structured SQL Code Generation & Self-Healing
# Why this suite of SQL Prompts?
# SQL generation by LLMs is highly prone to syntactic differences, missing joins, and 
# column hallucinations. To guarantee 100% valid execution, we split prompts into two paths:
#
# 1. Simple SQL Path:
#    - `get_sql_generation_prompt`: Direct single-shot generation. It injects retrieved 
#      dynamic lessons to prevent past failures and enforces strict schema rules.
#    - `get_sql_healing_prompt`: The debugger. When execution fails, it receives the exact
#      SQL that errored and the database error traceback to isolate the correction.
# 2. Complex SQL Divide-and-Conquer Path:
#    - `get_decomposer_prompt`: The Planner/Manager. Decouples complex multi-join queries
#      into isolated single-table sub-tasks (Logic Islands) and generates an AST assembly 
#      join plan, avoiding cognitive load on a single generation.
#    - `get_worker_prompt`: The Isolated Worker. Receives exactly one sub-task and is
#      strictly instructed to output clean SQL for its subset tables while forcing join 
#      columns to remain visible in SELECT blocks.
# 3. Response Summarization (`get_sql_response_format_prompt`):
#    Translates database result sets into concise natural language summaries, strictly
#    abiding by formatting parameters (e.g. no backticks).
# ##############################################################################


# ### --- SQL GENERATION PROMPT --- ###

def get_sql_generation_prompt() -> ChatPromptTemplate:
    """
    Factory function for the primary SQL generation system prompt.
    
    Integrates dynamic systemic lessons and the pruned schema to guide the LLM.
    
    Returns:
        A compiled ChatPromptTemplate for generating database queries.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert for the 'Pagila' DVD rental database.

### SYSTEMIC LESSONS (PAST MISTAKES)
{lessons}
If you apply a lesson, explicitly state which one and why (briefly).

DATABASE SCHEMA:
{schema}

STRICT RULES:
1. Provide a PostgreSQL query to answer the user's question. Populate the 'sql' field of the structured output.
2. Answer ONLY the specific question asked. Do not include extra columns or data not requested.
3. Use standard PostgreSQL syntax.
4. Always limit results to a maximum of 50 unless requested otherwise.
5. NO HALLUCINATIONS: Do not assume columns exist (e.g., 'name', 'country', 'total_amount') unless you see them in the SCHEMA.
6. STRUCTURAL INTEGRITY: If you need to filter by 'country', you MUST join via 'address' -> 'city' -> 'country'.
7. STRING MATCHING: Always use case-insensitive matching (ILIKE) for user-provided names or locations unless specified otherwise.
8. If the question cannot be answered with the provided schema, set the 'sql' field to "I cannot find the relevant tables for this request."
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])


# ### --- SQL HEALING PROMPT --- ###

def get_sql_healing_prompt() -> ChatPromptTemplate:
    """
    Factory function for the self-healing SQL debugger prompt.
    
    Injects the exact failed statement and database engine error trace for debugging.
    
    Returns:
        A compiled ChatPromptTemplate for self-healing SQL debug operations.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL debugging expert. 
A previous SQL query failed. FIX it based on the error and schema.

DATABASE SCHEMA:
{schema}

FAILED QUERY: {failed_query}
ERROR MESSAGE: {error_message}

INSTRUCTIONS:
- Provide a CORRECTED PostgreSQL query that answers ONLY the user's question.
- ONLY return the SQL query. No explanation.
"""),
        ("human", "Fix the query for: {question}")
    ])


# ### --- DECOMPOSER PROMPT --- ###

def get_decomposer_prompt() -> ChatPromptTemplate:
    """
    Factory function for the Manager node query decomposition prompt.
    
    Splits complex multi-table queries into atomic logic islands and a merge join blueprint.
    
    Returns:
        A compiled ChatPromptTemplate for multi-agent decomposition.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL Strategy Manager for the Pagila (DVD Rental) database.
Your goal is to decompose a complex natural language question into atomic sub-tasks (Logic Islands) AND a deterministic join plan.

SKELETON SCHEMA (Tables & FKs):
{skeleton_schema}

### STRATEGY:
1. Divide the question into 'Islands of Logic' (e.g., Filtering by Category vs Filtering by Actor).
2. Each 'SubTask' must be atomic and include columns needed for the FINAL join.
3. The 'JoinPlan' is MANDATORY. It must be a step-by-step blueprint to merge these islands.

### RULES:
- Use 'inner' joins by default.
- **JOIN KEY PROTECTION**: Ensure 'required_columns' includes the Join Key (e.g., 'film_id', 'customer_id').
- **NO HALLUCINATIONS**: Do not assume columns exist on a table just because they are related. 
  - Example: 'category' is NOT in the 'film' table. You MUST use 'film_category' and 'category' tables.
  - Example: 'revenue' is calculated from the 'payment' or 'rental' tables.
- Isolation: Each task should handle its own tables.
- You MUST provide both 'sub_tasks' AND 'join_plan'. Do not skip the join_plan.

### JOIN PLAN EXAMPLE:
- base_task: "task_1"
- steps: [ {{"left": "task_1", "right": "task_2", "on": "film_id"}} ]
- final_select: "task_1.title, task_2.revenue"
- order_by: "task_2.revenue DESC"
- limit: 5
"""),
        ("human", "{question}")
    ])


# ### --- WORKER PROMPT --- ###

def get_worker_prompt() -> ChatPromptTemplate:
    """
    Factory function for the isolated worker node query generation prompt.
    
    Forces generation of a singular, self-contained query snippet targeting specific tables.
    
    Returns:
        A compiled ChatPromptTemplate for SQL worker generation.
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
- STRICT SCHEMA ADHERENCE: Do NOT use columns not listed in the SCHEMA for a given table.
"""),
        ("human", "TASK: \"{task_description}\"")
    ])


# ### --- RESPONSE FORMAT PROMPT --- ###

def get_sql_response_format_prompt() -> ChatPromptTemplate:
    """
    Factory function for summarizing raw SQL output into client-friendly context-aware answers.
    
    Enforces clean markdown formatting, strictly prohibiting formatting tricks like backticks.
    
    Returns:
        A compiled ChatPromptTemplate for database answer summarization.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a concise Data Analyst for the 'Pagila' DVD rental store.
Your goal is to provide a natural language 'summary' of the aggregated database result.

USER QUESTION: {question}
SQL EXECUTED: {query}
AGGREGATED RESULT: {data}

STRICT INSTRUCTIONS FOR THE SUMMARY:
1. Provide a single, natural, context-aware sentence that directly states the aggregated value answering the user's question (e.g., 'The total rental revenue generated by Mary Smith is $118.68.').
2. Use standard markdown bolding (`**`) for key highlights like customer names or specific values.
3. DO NOT use backticks (`` ` ``) anywhere in the summary under any circumstances (never wrap names, values, or numbers in backticks).
4. Be concise and professional. Do not volunteer extra analysis or unasked tips.
5. DO NOT include the SQL query inside the summary.
"""),
        ("human", "Summarize these results.")
    ])

