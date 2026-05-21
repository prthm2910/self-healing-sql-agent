from langchain_core.prompts import ChatPromptTemplate

def get_lesson_distillation_prompt():
    """
    Lesson distillation system prompt.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Staff Engineer mentoring junior agents.
Create a "Golden standard Lesson" from this SQL mistake. Output MUST be valid JSON with keys: "is_global", "tags", "title", "body", "ending_note", "thought_process".
FAIL: {failed_sql}
ERROR: {sql_error}
FIX: {fixed_sql}
""")
    ])
