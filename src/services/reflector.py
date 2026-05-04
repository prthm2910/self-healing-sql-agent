import hashlib
from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from src.core.config import settings
from src.services.llm import get_chat_model
from src.prompts.reflector import get_reflector_prompt
from src.utils.logger import logger


class Fact(BaseModel):
    """A single persistent fact about the user."""
    fact: str = Field(description="The core fact or preference extracted.")
    category: str = Field(description="Category of the fact.")
    certainty: float = Field(description="Confidence score from 0 to 1.")


class Facts(BaseModel):
    """A collection of extracted facts."""
    facts: List[Fact]


def extract_and_store_facts(history: list, user_id: str, store: BaseStore):
    """Extracts facts from history and puts them into the PostgresStore."""
    logger.info(f"Starting reflection for User {user_id} with {len(history)} messages.")
    
    # 1. Fetch current memories to prevent redundant extraction
    # We search the store for existing memories in this user's namespace.
    try:
        existing_memories = store.search((user_id, "memories"), limit=100)
        formatted_existing = "\n".join([f"- {m.value['fact']}" for m in existing_memories]) if existing_memories else "None"
    except Exception as e:
        logger.error(f"Could not fetch existing memories for reflection context: {e}")
        formatted_existing = "None"
    
    # 2. Initialize modern LCEL chain
    try:
        prompt_template = get_reflector_prompt()
        llm = get_chat_model(is_flash=True).with_structured_output(Facts)
        chain = prompt_template | llm
        
        # 3. Format history for the prompt
        formatted_history = ""
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_history += f"{role}: {msg.content}\n"
        
        # 4. Extract facts using the template
        logger.debug("Invoking LLM for fact extraction...")
        extraction = chain.invoke({
            "history": formatted_history,
            "existing_memories": formatted_existing
        })

        # 5. Store each fact
        if extraction and extraction.facts:
            logger.info(f"Extracted {len(extraction.facts)} facts.")
            for fact_item in extraction.facts:
                # Deterministic ID based ONLY on normalized fact content.
                # This ensures category changes don't create duplicates.
                normalized_fact = fact_item.fact.lower().strip()
                fact_hash = hashlib.md5(normalized_fact.encode()).hexdigest()
                fact_id = fact_hash # Content-primary ID

                logger.debug(f"Storing fact {fact_id}: {fact_item.fact}")
                store.put(
                    (user_id, "memories"),
                    fact_id,
                    fact_item.model_dump()
                )
                logger.info(f"Stored memory: {fact_item.fact}")
        else:
            logger.info("No new facts extracted from this conversation.")
            
    except Exception as e:
        logger.error(f"Error in background reflection: {e}", exc_info=True)
