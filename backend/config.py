from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    max_articles_per_source: int = 25
    dedup_similarity_threshold: float = 0.85
    relevance_similarity_threshold: float = 0.35
    hours_lookback: int = 24

    synthesis_model: str = "claude-sonnet-4-6"
    context_model: str = "claude-haiku-4-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
