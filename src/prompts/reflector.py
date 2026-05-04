from langchain_core.prompts import ChatPromptTemplate


def get_reflector_prompt():
    """
    Factory function to return the Memory Reflector prompt template.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Memory Reflector. Your job is to analyze a conversation "
            "and extract persistent facts, preferences, or important details "
            "about the user.\n\n"
            "Rules:\n"
            "1. ATOMIC EXTRACTION: Extract ONE discrete 'Fact' object per individual piece of information. "
            "NEVER summarize, merge, or group distinct facts. Even if they are related (e.g., 'likes red' and 'likes blue'), "
            "they MUST remain separate facts.\n"
            "2. MEMORY AWARENESS: You will be provided with 'Existing Memories'. DO NOT extract information that is already "
            "present in the existing memories. Only extract NEW details.\n"
            "3. EXAMPLE: If the user says 'I like coffee and I like tea', and 'User likes coffee' is already in memories, "
            "you MUST only extract 'User likes tea'.\n"
            "4. STRICT RULE: NEVER extract SQL results, table data, or database records as memories. "
            "Only extract personal user facts (name, location, preferences) or meta-lessons about how to interact with the user.\n"
            "5. ONLY extract facts that are likely to be useful in future conversations.\n"
            "6. If the user expresses a contradiction (e.g., they liked tea before "
            "but now say they like coffee), extract the NEW fact.\n"
            "7. Output MUST be valid JSON matching the provided schema."
        ),
        (
            "user", 
            "Existing Memories:\n{existing_memories}\n\n"
            "Conversation History:\n{history}\n\n"
            "Extract any NEW facts from the conversation above that are not in the existing memories."
        ),
    ])
