from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # Section 4.5 / 25: session for the public demo (per-browser, in-memory,
    # resets on refresh), sqlite for your own local, persistent instance.
    portfolio_backend: Literal["sqlite", "session"] = "sqlite"
    database_url: str = "sqlite:///./quantpulse.db"

    # Section 4.3: LLM is a narrator over precomputed numbers, never the
    # source of them. Swappable via this flag; the app works with it unset.
    llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    # Section 5: free-tier data source credentials.
    finnhub_api_key: str | None = None
    fred_api_key: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None

    # SEC requires a descriptive User-Agent (name + contact email) on every
    # EDGAR request, or it will throttle/reject the request.
    sec_edgar_user_agent: str = "QuantPulse research contact-not-set@example.com"

    # Section 6.5/6.6: on-disk response cache for ingestion clients.
    ingestion_cache_dir: str = ".cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
