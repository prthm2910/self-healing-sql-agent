# ### --- IMPORTS --- ###
from langchain_core.prompts import ChatPromptTemplate

# ##############################################################################
# [Elaborative Breakdown] Context-Aware Security Guardian & Clarification Locks
# Why a split Guardian structure?
# In transactional user-facing AI chat applications, users frequently provide ambiguous,
# brief, or vague inputs (e.g., "show films", "help me"). Generating SQL for such queries
# directly can result in massive, slow database queries or irrelevant results.
#
# Our system uses a two-tiered "Clarification Lock" pattern:
# 1. Unlocked Guardian (`get_unlocked_guardian_prompt`):
#    Evaluates new user messages to check if they have valid database query intents (SQL), 
#    are general chit-chat / unauthorized requests (DENY), or are related but vague (CLARIFY).
# 2. Locked Guardian (`get_locked_guardian_prompt`):
#    If the previous interaction resulted in a CLARIFY status, the state enters a lock, 
#    recording the vague prompt context. When the user responds to our clarification 
#    question, the Locked Guardian evaluates their message *in the context of* the 
#    original vague request to resolve it. This provides continuity and ensures the user
#    remains in the loop until a concrete query intent is clarified.
# ##############################################################################


# ### --- LOCKED GUARDIAN FACTORY --- ###

def get_locked_guardian_prompt() -> ChatPromptTemplate:
    """
    Factory function for the Context-Aware Locked Guardian prompt template.
    
    Used when the conversation is locked in a clarification loop, evaluating the new
    user response alongside the cached vague question context.
    
    Returns:
        A compiled ChatPromptTemplate for locked context validation.
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


# ### --- UNLOCKED GUARDIAN FACTORY --- ###

def get_unlocked_guardian_prompt() -> ChatPromptTemplate:
    """
    Factory function for the standard Unlocked Guardian prompt template.
    
    Performs primary intent classification (SQL, DENY, or CLARIFY) on fresh user messages.
    
    Returns:
        A compiled ChatPromptTemplate for standard intent classification.
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


# ### --- CLARIFICATION PROMPT FACTORY --- ###

def get_clarification_prompt() -> ChatPromptTemplate:
    """
    Factory function for generating dynamic customer clarification questions.
    
    Used to prompt the user for specific parameters when their query is classified as CLARIFY.
    
    Returns:
        A compiled ChatPromptTemplate for generating clarification questions.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "The user asked: '{last_msg}'. This is too vague for a SQL query. Ask a concise follow-up question to clarify what they want to see from the DVD rental database. Output MUST be valid JSON with key: 'clarification_question'.")
    ])

