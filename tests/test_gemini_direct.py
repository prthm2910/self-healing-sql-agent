from src.services.llm import get_llm
from langchain_core.messages import HumanMessage
from src.utils.logger import logger

def test_gemini_inference():
    print("Testing Google Gemini Inference...")
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Hello, say 'Gemini is active' if you can hear me.")])
        print(f"Content Type: {type(response.content)}")
        print(f"Response Content: {response.content}")
        
        content = response.content
        if isinstance(content, list):
            # Extract text from the list of blocks
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            content = "".join(text_parts)
            print(f"Extracted Text: {content}")

        if "Gemini is active" in content:
            print("SUCCESS: Google Gemini API is working correctly.")
        else:
            print("WARNING: Received unexpected response format.")
    except Exception as e:
        print(f"FAILURE: Google Gemini API error: {e}")

if __name__ == "__main__":
    test_gemini_inference()
