from typing import List, Dict, Any
from src.utils.logger import logger

def get_relevant_lessons(query: str, store, selected_tables: List[str] = None, limit_dynamic: int = 3, limit_global: int = 5) -> tuple[str, list[str]]:
    """
    Retrieves Tiered Lessons:
    1. Global (Pinned)
    2. Table-Specific (Based on selected_tables tags)
    3. Semantic (Match to query)
    """
    if store is None:
        return "", []

    try:
        all_lessons_text = []
        applied_titles = []
        
        # 1. Fetch Global Lessons
        global_results = store.search(("global", "lessons", "pinned"), query=None, limit=limit_global)
        if global_results:
            all_lessons_text.append("### GLOBAL RULES:")
            for res in global_results:
                title = res.value.get('title', 'Global')
                applied_titles.append(f"GLOBAL: {title}")
                all_lessons_text.append(f"- {res.value['instruction']} (Rule: {title})")

        # 2. Fetch Table-Specific Lessons (Exact tag match)
        if selected_tables:
            for table in selected_tables:
                # Assuming we store table names in a 'tags' list in the value
                # We can't do exact filter in search easily without specific fields indexed
                # For now, we'll do a semantic search with the table name as query in dynamic
                table_results = store.search(("global", "lessons", "dynamic"), query=f"table:{table}", limit=2)
                if table_results:
                    all_lessons_text.append(f"\n### EXPERTISE FOR TABLE '{table}':")
                    for res in table_results:
                        title = res.value.get('title')
                        applied_titles.append(f"TABLE({table}): {title}")
                        all_lessons_text.append(f"- {res.value['instruction']}")

        # 3. Fetch Semantic Lessons (Query match)
        semantic_results = store.search(("global", "lessons", "dynamic"), query=query, limit=limit_dynamic)
        if semantic_results:
            all_lessons_text.append("\n### RELEVANT PAST MISTAKES:")
            for res in semantic_results:
                title = res.value.get('title')
                if f"TABLE" not in str(applied_titles): # Avoid duplicates
                    applied_titles.append(f"SEMANTIC: {title}")
                    all_lessons_text.append(f"- {res.value['instruction']} (Context: {title})")

        formatted_text = "\n".join(all_lessons_text) if all_lessons_text else "No previous lessons found."
        return formatted_text, applied_titles
        
    except Exception as e:
        logger.error(f"Failed to retrieve lessons: {e}")
        return "", []

def record_lesson(title: str, mistake: str, instruction: str, reasoning: str, store, is_global: bool = False, tags: List[str] = None):
    """
    Records a lesson with optional table tags.
    """
    import uuid
    namespace = ("global", "lessons", "pinned") if is_global else ("global", "lessons", "dynamic")
    lesson_id = str(uuid.uuid4())
    
    lesson_data = {
        "title": title,
        "mistake": mistake,
        "instruction": instruction,
        "reasoning": reasoning,
        "tags": tags or []
    }
    store.put(namespace, lesson_id, lesson_data)
    logger.info(f"Recorded {'GLOBAL' if is_global else 'DYNAMIC'} lesson: {title} | Tags: {tags}")

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
