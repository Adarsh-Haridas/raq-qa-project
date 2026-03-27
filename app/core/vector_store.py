from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams,Distance

from app.utils.logger import get_logger
from app.config import get_settings
from app.core.embedding import get_embeddings

logger=get_logger(__name__)
settings=get_settings()
EMBEDDING_DIMENSION=3072

@lru_cache
def get_qdrant_client()->QdrantClient:

    logger.info(f"Connecting to QDRANT at {settings.qdrant_url}")

    client=QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
        
    )

    logger.info(f"Qdrant client connected Successfully")

    return client

class VectorStoreService :

    def __init__(self,collection_name : str | None = None):
        self.collection_name=collection_name or settings.collection_name
        self.client=get_qdrant_client()
        self.embeddings=get_embeddings()

        self.ensure_collection()

        self.vector_store=QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )

        logger.info(f"Vectorstore Initialized for collection: {self.collection_name}")

    def ensure_collection(self) -> None:
        try:
            collection_info=self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection: {self.collection_name} exists with "
                f"{collection_info.points_count} points"
            )        

        except UnexpectedResponse:
            logger.info(f"Creating collections: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                )
            )

            logger.info(f"Collection {self.collection_name} created Successfully!!")

    def add_documents(self,documents: list[Document]) -> list[str]:

        if not documents:
            logger.warning("No Documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        ids=[str(uuid4()) for _ in documents]

        self.vector_store.add_documents(documents=documents,ids=ids)
        logger.info(f"Successfully added {len(documents)} documents")
        return ids
    
    def search(self,query: str, k: int | None=None) -> list[Document]:

        k=k or settings.retrieval_k

        logger.debug(f"Searching for: {query[:50]}....k={k}")
        results=self.vector_store.similarity_search(query,k=k)
        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_with_score(self,query: str,k:int|None=None)->list[tuple[Document,float]]:
        k=k or settings.retrieval_k
        logger.debug(f"Searching with score for: {query[:50]}... k={k} ")
        results=self.vector_store.similarity_search_with_score(query,k=k)
        logger.debug(f"Found {len(results)} results with score")
        return results
    
    def get_retriever(self,k:int | None=None)-> Any:
        k=k or settings.retrieval_k
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":k}
        )
    
    def delete_collection(self) -> None:
        logger.warning(f"Deleting collection {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def get_collection_info(self) ->dict:

        try:
            info=self.client.get_collection(self.collection_name)
            return {
                "name":self.collection_name,
                "points_count":info.points_count,
                "indexed_vectors_count":info.indexed_vectors_count,
                "status":info.status
            }
        
        except UnexpectedResponse:
            return {
                "name":self.collection_name,
                "points_count":0,
                "indexed_vectors_count":0,
                "status":"Not Found"
            }
        
    def health_check(self)->None:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
