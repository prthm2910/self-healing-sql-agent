from typing import List, Dict, Any
from src.utils.logger import logger

def get_relevant_lessons(query: str, store, limit_specific: int = 2, limit_global: int = 3) -> tuple[str, list[str]]:
    """
    Retrieves both Global (Pinned) and Specific (Semantic) lessons.
    Returns: (formatted_text, applied_titles)
    """
    if store is None:
        return "", []

    try:
        # 1. Fetch Global Lessons (Rules that always apply)
        global_results = store.search(("global", "lessons", "pinned"), query=None, limit=limit_global)
        
        # 2. Fetch Specific Lessons (Semantic match to current query)
        specific_results = store.search(("global", "lessons", "dynamic"), query=query, limit=limit_specific)
        
        all_lessons_text = []
        applied_titles = []
        
        # Format Global
        if global_results:
            all_lessons_text.append("### GLOBAL RULES (Always Apply):")
            for res in global_results:
                title = res.value.get('title', 'Global Lesson')
                applied_titles.append(f"GLOBAL: {title}")
                all_lessons_text.append(f"- {res.value['instruction']} (Reason: {res.value['reasoning']})")
        
        # Format Specific
        if specific_results:
            all_lessons_text.append("\n### CONTEXT-SPECIFIC LESSONS:")
            for res in specific_results:
                title = res.value.get('title', 'Specific Lesson')
                applied_titles.append(f"SPECIFIC: {title}")
                all_lessons_text.append(f"- {res.value['instruction']} (Applied because your query relates to: {title})")

        formatted_text = "\n".join(all_lessons_text) if all_lessons_text else "No previous lessons found."
        return formatted_text, applied_titles
        
    except Exception as e:
        logger.error(f"Failed to retrieve lessons: {e}")
        return "", []

def record_lesson(title: str, mistake: str, instruction: str, reasoning: str, store, is_global: bool = False):
    """
    Records a lesson into either the 'pinned' or 'dynamic' namespace.
    """
    import uuid
    namespace = ("global", "lessons", "pinned") if is_global else ("global", "lessons", "dynamic")
    lesson_id = str(uuid.uuid4())
    
    lesson_data = {
        "title": title,
        "mistake": mistake,
        "instruction": instruction,
        "reasoning": reasoning
    }
    store.put(namespace, lesson_id, lesson_data)
    logger.info(f"Recorded {'GLOBAL' if is_global else 'DYNAMIC'} lesson: {title}")

def list_all_lessons(store) -> List[Dict[str, Any]]:
    """Fetches every lesson from both pinned and dynamic namespaces."""
    if store is None:
        return []
    
    all_lessons = []
    try:
        # Fetch from both namespaces
        for sub_type in ["pinned", "dynamic"]:
            results = store.search(("global", "lessons", sub_type), limit=100)
            for res in results:
                all_lessons.append({
                    "id": res.key,
                    "type": sub_type,
                    "title": res.value.get("title"),
                    "mistake": res.value.get("mistake"),
                    "instruction": res.value.get("instruction"),
                    "reasoning": res.value.get("reasoning")
                })
    except Exception as e:
        logger.error(f"Failed to list lessons: {e}")
    return all_lessons

def delete_lesson(store, lesson_id: str, lesson_type: str):
    """Deletes a specific lesson."""
    if store is None:
        return
    try:
        namespace = ("global", "lessons", lesson_type)
        store.delete(namespace, lesson_id)
        logger.info(f"Deleted lesson: {lesson_id} from {lesson_type}")
    except Exception as e:
        logger.error(f"Failed to delete lesson: {e}")
