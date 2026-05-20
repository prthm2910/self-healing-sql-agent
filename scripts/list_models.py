from google import genai
from src.core.config import settings

def list_available_models():
    client = genai.Client(api_key=settings.google_api_key)
    print("Available models:")
    try:
        # The new SDK has a different way to list models
        for m in client.models.list():
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_available_models()
