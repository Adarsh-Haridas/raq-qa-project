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

    #Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api: str

    #Collection Settings
    collection_name: str="rag_documents"

    #Document Processing settings
    chunk_size: int=1000
    chunk_overlap: int=200


@lru_cache
def get_settings() -> Settings:
    return Settings()



