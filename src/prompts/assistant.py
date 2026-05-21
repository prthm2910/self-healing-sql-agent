# ### --- IMPORTS --- ###
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ##############################################################################
# [Elaborative Breakdown] Prompt Engineering & Tiered Retrieval Architecture
# Why structured system instruction templates?
# In LLM-powered agentic systems, prompt composition directly impacts instruction
# adherence, structural consistency, and context windows. We encapsulate our prompts
# inside dedicated template factory functions.
#
# Key Strategies:
# 1. Tiered Retrieval Hierarchy:
#    By embedding dynamic `{lessons}` directly within the system block, we prime the LLM's
#    in-context memory with relevant historical SQL errors and their fixes before the 
#    conversational history is evaluated. This minimizes cognitive load during message 
#    generation and ensures lessons override generic behaviors.
# 2. Strict XML/Markdown Sectioning:
#    We organize instructions using clear markdown header sectioning (e.g. `### SYSTEMIC LESSONS`).
#    This leverages the LLM's structural parser to clearly segment metadata from operational
#    rules.
# ##############################################################################


# ### --- ASSISTANT PROMPT FACTORY --- ###

def get_assistant_prompt() -> ChatPromptTemplate:
    """
    Factory function to construct the personalized AI Assistant prompt template.
    
    Includes injected systemic lessons derived from historical database execution failures
    integrated with historical chat messaging context.
    
    Returns:
        A compiled ChatPromptTemplate containing the system prompt and history placeholders.
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

