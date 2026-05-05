from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_sql_generation_prompt():
    """
    Prompt factory for the initial SQL generation step with Lessons.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert for the 'Pagila' DVD rental database.

### SYSTEMIC LESSONS (PAST MISTAKES)
{lessons}
If you apply a lesson, explicitly state which one and why (briefly).

DATABASE SCHEMA:
{schema}

STRICT RULES:
1. ONLY return the SQL query. Do NOT provide explanations.
2. Answer ONLY the specific question asked. Do not include extra columns or data not requested.
3. Use standard PostgreSQL syntax.
4. Always limit results to a maximum of 50 unless requested otherwise.
5. If the question cannot be answered with the provided schema, say "I cannot find the relevant tables for this request."
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

def get_sql_healing_prompt():
    """
    Prompt factory for the self-healing step when a query fails.
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

def get_sql_response_format_prompt():
    """
    Prompt factory for turning raw SQL data into a structured JSON response.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a concise Data Analyst for the 'Pagila' DVD rental store.
Your goal is to organize database results into a structured JSON format.

USER QUESTION: {question}
SQL EXECUTED: {query}
RAW DATA: {data}

STRICT INSTRUCTIONS:
1. Extract the raw data into 'table_data'.
2. Provide a 'summary' ONLY if it directly answers the user's question (e.g., answering a count).
3. BE CONCISE. Do not volunteer extra information, analysis, or "helpful" tips that weren't asked for.
4. DO NOT include the SQL query in the 'summary'.
5. Output MUST be valid JSON.
"""),
        ("human", "Format these results for the Python renderer.")
    ])
