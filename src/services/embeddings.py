# ### --- IMPORTS --- ###
import time
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import settings
from src.utils.logger import logger

# ##############################################################################
# [Elaborative Breakdown] Google Generative AI Embeddings & Batch Workaround
# Why LoggedEmbeddings & Batch Workaround?
# The standard `langchain-google-genai` package has a known batching defect where
# `embed_documents` (designed to generate vector representations for an array of 
# documents in one API request) occasionally wraps or truncates the response, returning
# only a single embedding vector instead of a list of vectors matching the input size.
#
# Workaround:
# We override `embed_documents` to resolve each text block independently via
# `super().embed_query` sequentially.
#
# Trade-offs:
# 1. Performance: Sequentially making single embedding generation API calls adds network
#    round-trip overhead compared to a single bulk batch request. However, since document
#    ingestion and semantic lookup in our self-healing assistant occur in highly targeted,
#    low-cardinality blocks (usually 1-5 entries), the resilience benefit outweighs the 
#    overhead.
# 2. Strict Logging: Every embedding execution is wrapped in precise timing blocks to 
#    assist in backend tracing and identify latency bottlenecks in production.
# ##############################################################################


# ### --- EMBEDDINGS CLASS --- ###

class LoggedEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Custom Google Generative AI Embeddings provider wrapped with latency and batch-resilience logging.
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of documents sequentially to avoid LangChain library bugs.
        
        Args:
            texts: A list of string documents/chunks to generate embedding vectors for.
            
        Returns:
            A list of lists of floats, representing the high-dimensional embedding vectors.
            
        Raises:
            Exception: Re-raises any error occurring during the embedding generation process.
        """
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        start_time: float = time.time()
        try:
            # Sequential resolution workaround for langchain-google-genai batching bug
            result: List[List[float]] = [super().embed_query(text) for text in texts]
            duration: float = time.time() - start_time
            logger.info(f"Generated {len(texts)} embeddings in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a single high-dimensional embedding vector for a query string.
        
        Args:
            text: The raw query string to convert into an embedding.
            
        Returns:
            A list of floats representing the embedding vector.
            
        Raises:
            Exception: Re-raises any error occurring during the embedding generation process.
        """
        logger.debug("Generating embedding for query...")
        start_time: float = time.time()
        try:
            result: List[float] = super().embed_query(text)
            duration: float = time.time() - start_time
            logger.debug(f"Generated query embedding in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            raise


# ### --- FACTORY FUNCTIONS --- ###

def get_embeddings_provider() -> LoggedEmbeddings:
    """
    Factory function to instantiate the customized LoggedEmbeddings provider.
    
    Returns:
        An instantiated and configured LoggedEmbeddings object.
    """
    logger.debug(f"Instantiating GoogleGenerativeAIEmbeddings (model: {settings.embedding_model})")
    return LoggedEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimensions
    )

