# ### --- [IMPORTS & CONFIGURATION] --- ###
import uuid
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import logger


# ### --- [TIERED COGNITIVE MEMORY SYSTEM] --- ###

# [Elaborative Breakdown]
# Tiered Retrieval-Augmented Memory (Tiered RAG):
# To prevent the LLM from repeating syntactical or logical database errors, we implement
# a three-tiered semantic memory architecture using PostgreSQL (pgvector).
#
# 1. Tier 1 - Global Pinned Rules: Strict database-wide directives (e.g., dialect syntax rules,
#    mandatory aggregate conventions) fetched statically for every execution context.
# 2. Tier 2 - Table-Specific Guidelines: Exact-tag filtered lessons retrieved dynamically
#    based on active schemas (e.g., specific rules for partitioned transactional customer tables).
# 3. Tier 3 - Semantic Historical Lessons: Dynamic long-term memory matching. Operates by
#    projecting the user query into embedding vector space to locate past execution failures,
#    the database exception caught, and the exact SQL modification that healed it.
#
# This structural RAG feedback loop dramatically increases agent resiliency over time,
# essentially caching execution experience.


# ### --- [LESSON RETRIEVAL SERVICES] --- ###

def get_relevant_lessons(
    query: str,
    store: Any,
    selected_tables: Optional[List[str]] = None,
    limit_dynamic: int = 3,
    limit_global: int = 5
) -> Tuple[str, List[str]]:
    """Retrieves and formats tiered lessons from pinned, table-specific, and semantic stores.

    Args:
        query: The user query string to match semantically.
        store: The active vector-backed pgvector checkpoint memory store.
        selected_tables: Table names currently checked out for this SQL query.
        limit_dynamic: Max semantic/table results to fetch.
        limit_global: Max pinned global guidelines to load.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - The formatted Markdown instructions block ready to inject into the LLM context.
            - A list of applied lesson titles for trace tracking and debugging.
    """
    if store is None:
        return "", []

    try:
        all_lessons_text: List[str] = []
        applied_titles: List[str] = []
        
        # 1. Fetch Pinned Global Lessons: Loads general PostgreSQL rules (e.g. JOIN syntaxes) statically.
        # Query: None (loads all matching records without vector distance check).
        global_results = store.search(("global", "lessons", "pinned"), query=None, limit=limit_global)
        if global_results:
            all_lessons_text.append("### GLOBAL RULES:")
            for res in global_results:
                title = res.value.get("title", "Global")
                applied_titles.append(f"GLOBAL: {title}")
                # Inject concrete instruction into context block
                all_lessons_text.append(f"- {res.value['instruction']} (Rule: {title})")

        # 2. Fetch Table-Specific Lessons (Dynamic Table Schema Match):
        # Filters database lessons using table name tag metadata tags, shielding the model from unrelated table context.
        if selected_tables:
            for table in selected_tables:
                # Semantic search using "table:{name}" keyword parameters to locate custom expert constraints.
                table_results = store.search(("global", "lessons", "dynamic"), query=f"table:{table}", limit=2)
                if table_results:
                    all_lessons_text.append(f"\n### EXPERTISE FOR TABLE '{table}':")
                    for res in table_results:
                        title = res.value.get("title", "Table Expert")
                        applied_titles.append(f"TABLE({table}): {title}")
                        all_lessons_text.append(f"- {res.value['instruction']}")

        # 3. Fetch Semantic Lessons (Direct Query-to-Mistake Distance Match):
        # Employs cosine distance vector similarity search on the user's natural query.
        semantic_results = store.search(("global", "lessons", "dynamic"), query=query, limit=limit_dynamic)
        if semantic_results:
            all_lessons_text.append("\n### RELEVANT PAST MISTAKES:")
            for res in semantic_results:
                title = res.value.get("title", "Mistake")
                # Cognitive Guard: Prevent duplicating lessons already matched by table-specific queries
                if "TABLE" not in str(applied_titles):
                    applied_titles.append(f"SEMANTIC: {title}")
                    all_lessons_text.append(f"- {res.value['instruction']} (Context: {title})")

        # Join the text lists using double-newlines for optimal prompt layout
        formatted_text: str = "\n".join(all_lessons_text) if all_lessons_text else "No previous lessons found."
        return formatted_text, applied_titles
        
    except Exception as e:
        logger.error(f"Failed to retrieve lessons: {e}", exc_info=True)
        return "", []


# ### --- [LESSON PERSISTENCE SERVICES] --- ###

def record_lesson(
    title: str,
    mistake: str,
    instruction: str,
    reasoning: str,
    store: Any,
    is_global: bool = False,
    tags: Optional[List[str]] = None
) -> None:
    """Persists a synthesized SQL lesson learned into the pgvector semantic store.

    Args:
        title: Short descriptive title of the learning (e.g., 'JSON Column Unnesting').
        mistake: The malformed SQL or action that triggered the failure.
        instruction: Concrete correction directive (e.g., 'Use JSONB_TO_RECORDSET').
        reasoning: The explanatory logic of why the correction works.
        store: The active vector-backed pgvector checkpoint memory store.
        is_global: If True, writes to the pinned namespace; otherwise the dynamic namespace.
        tags: List of related table names or category labels.
    """
    # Select target namespace tuple based on global status
    namespace: Tuple[str, str, str] = (
        ("global", "lessons", "pinned") if is_global else ("global", "lessons", "dynamic")
    )
    # Generate unique UUID to prevent key collisions in the state checkpoint store
    lesson_id: str = str(uuid.uuid4())
    
    # Package into a structured dictionary contract
    lesson_data: Dict[str, Any] = {
        "title": title,
        "mistake": mistake,
        "instruction": instruction,
        "reasoning": reasoning,
        "tags": tags or []
    }
    # Execute upsert in pgvector state checkpoint store
    store.put(namespace, lesson_id, lesson_data)
    logger.info(f"Recorded {'GLOBAL' if is_global else 'DYNAMIC'} lesson: {title} | Tags: {tags}")


# ### --- [LESSON UTILITIES] --- ###

def list_all_lessons(store: Any) -> List[Dict[str, Any]]:
    """Retrieves and packages every lesson recorded across pinned and dynamic stores.

    Used by dashboard UI components to present learning histories.

    Args:
        store: The active vector-backed pgvector checkpoint memory store.

    Returns:
        List[Dict[str, Any]]: List of unified, parsed lesson records.
    """
    if store is None:
        return []
    
    all_lessons: List[Dict[str, Any]] = []
    try:
        # Loop through both namespace categories sequentially
        for sub_type in ["pinned", "dynamic"]:
            # Query vector store with high limit (100) to populate historical listing page
            results = store.search(("global", "lessons", sub_type), limit=100)
            for res in results:
                # Package and append standardized record shapes
                all_lessons.append({
                    "id": res.key,
                    "type": sub_type,
                    "title": res.value.get("title"),
                    "mistake": res.value.get("mistake"),
                    "instruction": res.value.get("instruction"),
                    "reasoning": res.value.get("reasoning")
                })
    except Exception as e:
        logger.error(f"Failed to list lessons: {e}", exc_info=True)
    return all_lessons
