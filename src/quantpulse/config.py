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

    # Section 11's "everything else in the app works with the LLM provider
    # entirely turned off" as an explicit switch, so narration can be disabled
    # outright (e.g. on the public demo) without unsetting a working key.
    llm_enabled: bool = True
    # Free-tier model per provider (Section 4.3's own picks). Overridable
    # because provider model catalogs churn faster than this repo will.
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    # Narration is a paragraph, not an essay: a low cap keeps every call cheap
    # against a free-tier quota (Section 11) and keeps the UI readable. Low
    # temperature because the job is restating computed numbers faithfully,
    # not being creative about them.
    llm_max_output_tokens: int = 400
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30.0

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

    # Sections 5 & 22: point-in-time S&P 500 membership (incl. removed names)
    # so the cold-start backfill is survivorship-bias-aware. Defaults to a
    # public interval-format dataset (ticker,start_date,end_date); override
    # with a local path via historical_constituents_path, or blank the URL to
    # force the documented-limitation fallback to today's constituents only.
    historical_constituents_url: str = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
    )
    historical_constituents_path: str | None = None

    # Section 6.2: how far back the one-time cold-start backfill pulls prices.
    #
    # NOT "max", despite that being the obvious choice. Yahoo's endpoint serves
    # `period="max"` unreliably: on a paced test (5s apart, nothing else
    # running) it returned ZERO rows for AAPL, MSFT, KO and AMZN, while
    # `period="10y"` returned a full 2,512 bars for every one of them. The
    # first real cold-start run used "max" and finished with price history for
    # only 496 of 1,206 symbols -- including no data at all for Apple, Amazon
    # and Adobe -- which read as "these symbols have no history" rather than
    # "this period argument is broken".
    #
    # 10 years comfortably covers everything downstream: the strategy backtest
    # looks back 5 years, the longest forecast horizon is 252 trading days, and
    # the forecast price window is 1,280 days. It also roughly halves the
    # database, which matters because the demo DB is committed to git.
    seed_history_period: str = "10y"


@lru_cache
def get_settings() -> Settings:
    return Settings()
