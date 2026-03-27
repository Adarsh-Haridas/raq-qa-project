from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):
    model_config=SettingsConfigDict(
        env_file='.env',
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    #OpenAi Configuration
    openai_api_key: str
    gemini_api_key: str

    #Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api_key: str

    #Collection Settings
    collection_name: str="rag_documents"

    #Document Processing settings
    chunk_size: int=1000
    chunk_overlap: int=200

    #Model configuration
    embedding_model: str = "models/gemini-embedding-001"
    llm_model:str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.0

    #Retrievel Settings
    retrieval_k: int = 4

    #logging
    log_level : str="INFO"

    #RAGA Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None
    ragas_llm_temperature: float | None = None
    ragas_embedding_model: str | None = None

    #API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    #Application info
    app_name: str = "RAQ Q&A System"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()



