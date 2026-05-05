import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings
from src.utils.logger import logger

class LoggedEmbeddings(GoogleGenerativeAIEmbeddings):
    """Wrapped embeddings provider with logging."""
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        start_time = time.time()
        try:
            result = super().embed_documents(texts)
            duration = time.time() - start_time
            logger.info(f"Generated {len(texts)} embeddings in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise

    def embed_query(self, text: str) -> list[float]:
        logger.debug(f"Generating embedding for query...")
        start_time = time.time()
        try:
            result = super().embed_query(text)
            duration = time.time() - start_time
            logger.debug(f"Generated query embedding in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            raise

def get_embeddings_provider():
    """Factory to get the logged embedding provider."""
    logger.debug(f"Instantiating GoogleGenerativeAIEmbeddings (model: {settings.embedding_model})")
    return LoggedEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimensions
    )
