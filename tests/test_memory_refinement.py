import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from src.services.reflector import extract_and_store_facts, Facts, Fact

def test_atomic_fact_retention():
    """
    Test that similar facts are stored as distinct entries in the store
    and not merged by fuzzy matching.
    """
    # 1. Setup mocks
    mock_store = MagicMock()
    # Mock search to return empty list (no duplicates found) - though our new code doesn't call it
    mock_store.search.return_value = []
    
    mock_user_id = "test_user_123"
    mock_history = [
        HumanMessage(content="I like coffee. I also like tea. I really like water.")
    ]
    
    # 2. Mock the LLM chain to return three distinct facts
    mock_facts = Facts(facts=[
        Fact(fact="User likes coffee", category="preferences", certainty=1.0),
        Fact(fact="User likes tea", category="preferences", certainty=1.0),
        Fact(fact="User likes water", category="preferences", certainty=1.0)
    ])
    
    with patch("src.services.reflector.get_llm") as mock_get_model:
        # Create a mock that will be the result of get_llm().with_structured_output()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_facts

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain
        mock_get_model.return_value = mock_llm

        # We need to mock the pipe operator | because ChatPromptTemplate | MagicMock
        # might not work as expected in all LangChain versions
        with patch("src.services.reflector.get_reflector_prompt") as mock_get_prompt:
            mock_prompt = MagicMock()
            mock_prompt.__or__.return_value = mock_chain
            mock_get_prompt.return_value = mock_prompt

            # 3. Execute the function
            extract_and_store_facts(mock_history, mock_user_id, mock_store)        
        # 4. Verifications
        # Verify store.put was called 3 times
        assert mock_store.put.call_count == 3
        
        # Verify unique IDs were generated
        calls = mock_store.put.call_args_list
        ids = [call.args[1] for call in calls]
        assert len(set(ids)) == 3
        
        # Verify facts were stored correctly
        stored_facts = [call.args[2]["fact"] for call in calls]
        assert "User likes coffee" in stored_facts
        assert "User likes tea" in stored_facts
        assert "User likes water" in stored_facts
        
        print("\n✅ Verification passed: 3 atomic facts stored with unique IDs.")

if __name__ == "__main__":
    pytest.main([__file__])
