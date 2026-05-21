
# ### --- [IMPORTS] --- ###

from typing import Dict, Any, Optional

from src.utils.logger import logger
from src.services.lessons import record_lesson
from src.workflow.schema.lessons import LessonDistillationOutput
from src.workflow.nodes.base import BaseNode, llm
from src.prompts.lessons import get_lesson_distillation_prompt


# ### --- [BACKGROUND TASKS] --- ###

# [Elaborative Breakdown]
# The `background_distill_lesson` helper function implements an asynchronous, non-blocking 
# memory distillation pipeline.
#
# --- Why Background Offloading is Necessary ---
# Lesson distillation requires prompting the LLM to analyze the delta between a failed SQL 
# query (with its raw engine error message) and the successfully self-healed SQL query. This
# analysis incurs significant computational and API network latency (typically 1 to 3 seconds).
#
# --- Mechanics of Non-Blocking Thread Execution ---
# 1. Isolation: Instead of forcing the end-user to wait for the LLM to complete this analysis
#    during the active response rendering thread, the calling `HealSQLNode` spawns a new daemon
#    `threading.Thread` to execute this distillation asynchronously.
# 2. Resiliency and Safety: Since the background thread is isolated, any network timeouts or
#    Groq/LLM failures inside the distillation pipeline are caught locally inside a global
#    try/except block, ensuring that failure to write a lesson never compromises or halts the
#    user's active query execution thread.
# 3. Persistent Storage Injection: Once the structured lesson is distilled, it is logged into
#    the vector/semantic store (`record_lesson`), categorizing it under its relevant tags and tables
#    for immediate lookup in future request planning.
def background_distill_lesson(state_data: Dict[str, Any], store: Any, user_id: str) -> None:
    """
    Background worker task to analyze SQL failures and distill key lessons.

    Extracts details about a failed query, maps it against the successfully healed query, 
    prompts the LLM for structured insights, and records the resulting lesson into the
    semantic memory lesson store.

    Args:
        state_data (Dict[str, Any]): Dictionary containing execution data:
            - `current_sql` (str): The failed query string.
            - `sql_error` (str): The raw database engine error message.
            - `fixed_sql` (str): The healed and verified query.
            - `selected_tables` (Optional[List[str]]): The tables referenced in the query.
        store (Any): The semantic lesson persistence database/store.
        user_id (str): The user identifier associated with the request context.

    Returns:
        None
    """
    try:
        # 1. Compile Distillation Prompt: Bind the raw error, failed query, and final fixed SQL statements.
        prompt_template = get_lesson_distillation_prompt()
        prompt_val = prompt_template.invoke({
            "failed_sql": state_data['current_sql'],
            "sql_error": state_data['sql_error'],
            "fixed_sql": state_data['fixed_sql']
        })
        
        # 2. Call Distillation Model: Invoke structured generator with robust regex-fallback parsing.
        distiller = llm.with_structured_output(LessonDistillationOutput)
        lesson: LessonDistillationOutput = BaseNode.robust_invoke(
            distiller, 
            prompt_val.to_messages(), 
            LessonDistillationOutput
        )
        
        # 3. Persist Lesson to pgvector: Save dynamic learning into our semantic database store.
        # Categorizes under table tags to ensure table-specific lookups can load it immediately next run.
        record_lesson(
            lesson.title, 
            state_data['sql_error'], 
            lesson.body.instruction, 
            lesson.body.reasoning, 
            store, 
            is_global=lesson.is_global,
            tags=state_data.get('selected_tables')
        )
    except Exception as e:
        # Isolated Error Handling: Catch exceptions locally to prevent background thread crashes
        # from muddying the main request execution loop or failing the response.
        logger.error(f"Background lesson distillation failed: {e}")


