from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.utils.logger import get_logger
from app.config import get_settings


logger=get_logger(__name__)

@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings=get_settings()
    logger.info(f"Initializing Embedding  model: {settings.embedding_model}")
    embeddings=GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.gemini_api_key
    )
    logger.info(f"Initialized Embedding Model: {settings.embedding_model}")
    return embeddings

class EmbeddingService:

    def __init__(self):
        settings=get_settings()
        self.embedding=get_embeddings()
        self.model_name=settings.embedding_model

    def embed_query(self,text :str) -> list[float]:

        logger.debug(f"Generating Embedding for query: {text[:50]}...")
        return self.embedding.embed_query(text)
    
    def embed_documents(self,texts: list[str]) -> list[list[float]]:
        logger.debug(f"Generating Embeddings for {len(texts)} documents")
        return self.embedding.embed_documents(texts)
