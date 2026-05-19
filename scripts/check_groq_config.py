from src.core.config import settings
from src.services.llm import get_llm

print(f"Model: {settings.model_name}")
print(f"Base URL: {settings.llm_base_url}")
print(f"Groq API Key set: {bool(settings.groq_api_key)}")
print(f"NVIDIA API Key set: {bool(settings.nvidia_api_key)}")

try:
    llm = get_llm()
    print("LLM Instantiation: SUCCESS")
except Exception as e:
    print(f"LLM Instantiation: FAILED ({e})")
