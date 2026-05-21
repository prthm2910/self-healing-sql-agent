import pytest
from unittest.mock import MagicMock, patch
from src.services.lessons import get_relevant_lessons
from src.workflow.nodes import call_chatbot
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

def test_get_relevant_lessons_transparency():
    """Verify get_relevant_lessons returns the new (text, titles) tuple."""
    mock_store = MagicMock()
    
    # Mocking search results
    mock_global = MagicMock()
    mock_global.value = {'title': 'Global Rule', 'instruction': 'Do X', 'reasoning': 'Because Y'}
    
    mock_specific = MagicMock()
    mock_specific.value = {'title': 'Pipe Mistake', 'instruction': 'Avoid |', 'reasoning': 'Syntax error'}
    
    # Configure mock search to return results for specific namespaces
    def side_effect(namespace, **kwargs):
        if namespace == ("global", "lessons", "pinned"):
            return [mock_global]
        if namespace == ("global", "lessons", "dynamic"):
            return [mock_specific]
        return []
    
    mock_store.search.side_effect = side_effect
    
    text, titles = get_relevant_lessons("how to use pipes?", mock_store)
    
    assert "### GLOBAL RULES" in text
    assert "### RELEVANT PAST MISTAKES" in text
    assert "GLOBAL: Global Rule" in titles
    assert "SEMANTIC: Pipe Mistake" in titles
    assert len(titles) == 2

def test_get_relevant_lessons_no_results():
    """Verify behavior when no lessons are found."""
    mock_store = MagicMock()
    mock_store.search.return_value = []
    
    text, titles = get_relevant_lessons("nothing", mock_store)
    
    assert text == "No previous lessons found."
    assert titles == []
    assert len(titles) == 0

@patch("src.workflow.nodes.get_llm")
@patch("src.workflow.nodes.get_assistant_prompt")
@patch("src.workflow.nodes.logger")
def test_chatbot_node_logging(mock_logger, mock_get_prompt, mock_get_model):
    """Verify the chatbot node logs the correct lesson count and titles."""
    from unittest.mock import patch
    
    mock_store = MagicMock()
    # Return 1 lesson
    mock_res = MagicMock()
    mock_res.value = {'title': 'Test Lesson', 'instruction': 'Keep it simple', 'reasoning': 'Logic'}

    def side_effect(ns, **kwargs):
        if len(ns) >= 3 and ns[2] == "pinned":
            return [mock_res]
        return []

    mock_store.search.side_effect = side_effect
    state = {"messages": [HumanMessage(content="Hello")]}
    config = {"configurable": {"user_id": "test_user"}}
    
    # Mock LLM response
    mock_chain = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    mock_get_prompt.return_value = mock_prompt
    
    call_chatbot(state, config, store=mock_store)
    
    # Check if logger.info was called with the expected string
    # "Chatbot Node | Memory: 0 | Lessons: 1 ['GLOBAL: Test Lesson']"
    log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any("Lessons: 1 ['GLOBAL: Test Lesson']" in s for s in log_calls)

if __name__ == "__main__":
    from unittest.mock import patch
    pytest.main([__file__])
