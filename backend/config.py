from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    max_articles_per_source: int = 15
    dedup_similarity_threshold: float = 0.85
    relevance_similarity_threshold: float = 0.30
    hours_lookback: int = 24

    # Haiku is ~5x faster than Sonnet and produces solid structured JSON for this task.
    # Override via SYNTHESIS_MODEL env var if you want higher-quality/slower Sonnet output.
    synthesis_model: str = "claude-haiku-4-5"
    context_model: str = "claude-haiku-4-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Cache a generated briefing for this many minutes (per focus). 0 disables caching.
    cache_ttl_minutes: int = 15
    # How many key developments to enrich with historical context (cap for speed).
    context_enrichment_limit: int = 4


settings = Settings()
