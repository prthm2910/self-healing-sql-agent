from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_sql_generation_prompt():
    """
    Prompt factory for the initial SQL generation step.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert for the 'Pagila' DVD rental database.
Your goal is to translate user questions into syntactically correct PostgreSQL queries.

DATABASE SCHEMA:
{schema}

RULES:
1. ONLY return the SQL query. Do NOT provide explanations.
2. Use standard PostgreSQL syntax.
3. Always limit results to a maximum of 50 unless the user asks for more.
4. If the question cannot be answered with the provided schema, say "I cannot find the relevant tables for this request."
5. Be careful with table joins. Use clear aliases.
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
A previous SQL query failed with an error. Your task is to FIX the query based on the error message and the schema.

DATABASE SCHEMA:
{schema}

FAILED QUERY:
{failed_query}

ERROR MESSAGE:
{error_message}

INSTRUCTIONS:
- Analyze the error (e.g., column doesn't exist, syntax error).
- Provide a CORRECTED PostgreSQL query.
- ONLY return the SQL query. No explanation.
"""),
        ("human", "Fix the query for: {question}")
    ])
