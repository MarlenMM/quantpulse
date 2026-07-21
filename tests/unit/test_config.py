from quantpulse.config import Settings, get_settings


def test_defaults_to_sqlite_portfolio_backend() -> None:
    settings = Settings(_env_file=None)
    assert settings.portfolio_backend == "sqlite"
    assert settings.database_url == "sqlite:///./quantpulse.db"


def test_portfolio_backend_accepts_session_mode() -> None:
    settings = Settings(_env_file=None, portfolio_backend="session")
    assert settings.portfolio_backend == "session"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
