from langchain_core.prompts import ChatPromptTemplate

def get_locked_guardian_prompt():
    """
    Context-aware locked guardian prompt.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Context-Aware Security Guardian.
The user previously made a vague request: "{vague_context}".
We asked for clarification. Their latest message is: "{last_msg}".

DOMAINS INFO:
{domain_summary}

### CLASSIFICATION RULES (Output JSON keys: "intent", "thought_process"):
- SQL: The new message, combined with the previous context, now forms a clear and valid SQL request about the domain.
- CLARIFY: The message is related but still vague. We stay in the clarification loop.
- DENY: The user has clearly changed the subject to something irrelevant OR is asking to modify data.

Note: If the user provides a COMPLETELY NEW but valid SQL request (e.g., pivoting from 'films' to 'customers'), classify as SQL and we will switch context.
""")
    ])

def get_unlocked_guardian_prompt():
    """
    Standard unlocked intent classification prompt.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are an Intent Classifier and Security Guardian.
Analyze the user message and determine if it relates to the database domain described below.

DOMAINS INFO:
{domain_summary}

### CLASSIFICATION RULES (Output JSON keys: "intent", "thought_process"):
- SQL: The request is a clear, actionable question about querying or analyzing data from the domain.
- CLARIFY: The request is related to the domain but is too vague, ambiguous, or underspecified to generate SQL for (e.g., "Tell me about films" without criteria).
- DENY: The request is NOT about the domain, asks to MODIFY data, or is general chitchat.

User Message: "{last_msg}"
""")
    ])

def get_clarification_prompt():
    """
    Follow-up clarification prompt.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "The user asked: '{last_msg}'. This is too vague for a SQL query. Ask a concise follow-up question to clarify what they want to see from the DVD rental database. Output MUST be valid JSON with key: 'clarification_question'.")
    ])
