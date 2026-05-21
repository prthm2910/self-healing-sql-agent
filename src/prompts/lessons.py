# ### --- IMPORTS --- ###
from langchain_core.prompts import ChatPromptTemplate

# ##############################################################################
# [Elaborative Breakdown] Self-Healing Memory & Golden Lesson Distillation
# Why Distillation?
# In multi-agent SQL systems, runtime execution failures (e.g., syntactical errors,
# missing joins, implicit schema hallucinations) are resolved by the self-healing loop
# at runtime. However, resolving the error once is not enough; without memory, the
# system would repeat the same error on subsequent runs.
#
# Golden Lesson Distillation:
# When the self-healing loop successfully debugs and executes a query, the Lesson
# Distiller (`get_lesson_distillation_prompt`) intercepts the transition. It acts
# like a Senior Staff Engineer conducting a Root Cause Analysis (RCA) by reviewing the 
# FAILED SQL, the error message, and the WORKING fix. It synthesizes a generalized,
# structured lesson containing an actionable instruction, which is then written to 
# the long-term `PostgresStore` vector index. Future agent invocations automatically
# retrieve and apply these lessons, preventing recurring mistakes.
# ##############################################################################


# ### --- LESSON DISTILLATION PROMPT FACTORY --- ###

def get_lesson_distillation_prompt() -> ChatPromptTemplate:
    """
    Factory function for the self-healing lesson distillation prompt template.
    
    Prompts the LLM to act as a Senior Staff Engineer, distilling a permanent lesson
    from a corrected query error to build long-term memory.
    
    Returns:
        A compiled ChatPromptTemplate for lesson distillation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Staff Engineer mentoring junior agents.
Create a "Golden standard Lesson" from this SQL mistake. Output MUST be valid JSON with keys: "is_global", "tags", "title", "body", "ending_note", "thought_process".
FAIL: {failed_sql}
ERROR: {sql_error}
FIX: {fixed_sql}
""")
    ])

