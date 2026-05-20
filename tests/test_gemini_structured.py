from typing import Literal
from pydantic import BaseModel, Field
from src.services.llm import get_llm
from src.utils.logger import logger

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(..., ge=0, le=1)

def test_gemini_structured():
    print("Testing Google Gemini Structured Output...")
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SentimentOutput)
        response = structured_llm.invoke("I absolutely love this new model, it's so fast!")
        print(f"Response: {response}")
        if response.sentiment == "positive":
            print("SUCCESS: Google Gemini Structured Output is working correctly.")
        else:
            print(f"WARNING: Unexpected sentiment: {response.sentiment}")
    except Exception as e:
        print(f"FAILURE: Google Gemini Structured Output error: {e}")

if __name__ == "__main__":
    test_gemini_structured()
