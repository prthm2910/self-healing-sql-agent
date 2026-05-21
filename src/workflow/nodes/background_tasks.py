
from src.utils.logger import logger
from src.services.lessons import record_lesson
from src.workflow.schema.lessons import LessonDistillationOutput
from src.workflow.nodes.base import BaseNode, llm
from src.prompts.lessons import get_lesson_distillation_prompt

def background_distill_lesson(state_data: dict, store, user_id: str):
    """Background task for lesson distillation."""
    try:
        prompt_template = get_lesson_distillation_prompt()
        prompt_val = prompt_template.invoke({
            "failed_sql": state_data['current_sql'],
            "sql_error": state_data['sql_error'],
            "fixed_sql": state_data['fixed_sql']
        })
        distiller = llm.with_structured_output(LessonDistillationOutput)
        lesson = BaseNode.robust_invoke(distiller, prompt_val.to_messages(), LessonDistillationOutput)
        
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
        logger.error(f"Background lesson distillation failed: {e}")
