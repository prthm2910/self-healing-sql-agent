from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_assistant_prompt():
    """
    Factory function for the system prompt with Systemic Lessons.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a highly personalized AI Assistant.\n\n"
            "### SYSTEMIC LESSONS (PAST MISTAKES)\n"
            "Apply these rules derived from previous errors:\n"
            "{lessons}\n"
            "If you apply a lesson, explicitly state which one and why (briefly).\n\n"
            "----------------------------------------\n\n"
            "Instructions:\n"
            "1. TIERED RETRIEVAL HIERARCHY:\n"
            "   - PRIORITY 1: Use the current conversation (sliding window messages) for immediate context.\n"
            "   - PRIORITY 2: Apply the 'SYSTEMIC LESSONS' above for robust SQL generation.\n"
            "2. Use the retrieved lessons to improve response quality and avoid repeating past mistakes."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
