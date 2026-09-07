from pathlib import Path

import pytest

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


class TestManualRefreshGate:
    """Who is allowed to start a refresh from the UI.

    Nothing refreshes on a schedule any more, so the button is the trigger --
    and a wrong default here is not cosmetic in either direction. Too strict and
    the only instance that can refresh its own data cannot; too loose and any
    visitor to the hosted demo can start a multi-hour job on shared free-tier
    quota, on a host whose `requirements.txt` omits the model stack it needs.
    """

    def test_a_local_sqlite_instance_may_refresh(self) -> None:
        assert Settings(_env_file=None).manual_refresh_allowed()

    def test_a_hosted_session_mode_demo_may_not(self) -> None:
        settings = Settings(_env_file=None, portfolio_backend="session")
        assert not settings.manual_refresh_allowed()

    def test_an_explicit_setting_overrides_the_inference(self) -> None:
        # Both directions: the inference is a default, not a policy.
        assert not Settings(_env_file=None, manual_refresh_enabled=False).manual_refresh_allowed()
        assert Settings(
            _env_file=None, portfolio_backend="session", manual_refresh_enabled=True
        ).manual_refresh_allowed()


class TestSchemaVersionGate:
    """A missed `alembic upgrade head` must fail in second one, not minute nine.

    It surfaced as `OperationalError: table composite_scores has no column named
    fundamental_raw` from whichever step wrote that column first -- which was
    after the run had already spent nine minutes fetching 503 tickers. The
    fetched data landed; the ratings table stayed empty.
    """

    @staticmethod
    def _engine_at(tmp_path: Path, revision: str, monkeypatch):
        """Migrate a temp database to `revision` and return an engine on it.

        Two alembic details matter here, and both bite silently:

        * `migrations/env.py` overrides `sqlalchemy.url` from `get_settings()`,
          so setting it on the Config has no effect -- the migration would run
          against the real database. The env var (plus clearing the `lru_cache`)
          is the only handle that redirects it.
        * The Config is built WITHOUT the ini path. `env.py` calls
          `fileConfig(config.config_file_name)` whenever one is present, and
          `fileConfig` defaults to `disable_existing_loggers=True` -- which
          disables every logger the app has already created, so unrelated tests
          later in the session silently stop capturing records via `caplog`.
          Passing `script_location` directly leaves `config_file_name` None and
          skips that entirely.
        """
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine

        from quantpulse.config import get_settings

        url = f"sqlite:///{tmp_path / f'{revision}.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        get_settings.cache_clear()
        root = Path(__file__).resolve().parents[2]
        config = Config()
        config.set_main_option(
            "script_location", str(root / "src" / "quantpulse" / "storage" / "migrations")
        )
        command.upgrade(config, revision)
        get_settings.cache_clear()
        return create_engine(url)

    def test_passes_on_a_database_at_head(self, tmp_path: Path, monkeypatch) -> None:
        from quantpulse.storage import db as db_module

        monkeypatch.setattr(db_module, "engine", self._engine_at(tmp_path, "head", monkeypatch))
        db_module.assert_schema_current()  # must not raise

    def test_raises_on_a_database_one_migration_behind(self, tmp_path: Path, monkeypatch) -> None:
        from quantpulse.storage import db as db_module

        monkeypatch.setattr(
            db_module, "engine", self._engine_at(tmp_path, "f7b3c4c4d737", monkeypatch)
        )
        with pytest.raises(db_module.SchemaOutOfDateError, match="alembic upgrade head"):
            db_module.assert_schema_current()

    def test_raises_on_a_completely_unmigrated_database(self, tmp_path: Path, monkeypatch) -> None:
        from sqlalchemy import create_engine

        from quantpulse.storage import db as db_module

        monkeypatch.setattr(
            db_module, "engine", create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
        )
        with pytest.raises(db_module.SchemaOutOfDateError):
            db_module.assert_schema_current()


class TestAlertingGate:
    """Whether the nightly refresh may send anything, and to where.

    The destination is a Discord webhook URL, which is a credential: whoever
    holds it can post into that channel. So it lives in a repo secret and
    reaches the process as an environment variable, and this repository is
    public -- an unset default that quietly does nothing is the only safe one.
    Every fork of this repo runs the refresh workflow with no secret set.
    """

    def test_alerting_is_off_until_a_destination_is_configured(self) -> None:
        assert Settings(_env_file=None).alert_discord_webhook_url is None
        assert not Settings(_env_file=None).alerting_configured()

    def test_a_configured_webhook_turns_it_on(self) -> None:
        settings = Settings(_env_file=None, alert_discord_webhook_url="https://example.test/hook")
        assert settings.alerting_configured()

    def test_the_kill_switch_works_without_unsetting_the_webhook(self) -> None:
        """Same shape as `llm_enabled`: turn the feature off without destroying
        a working credential you would then have to re-issue."""
        settings = Settings(
            _env_file=None,
            alert_discord_webhook_url="https://example.test/hook",
            alerts_enabled=False,
        )
        assert not settings.alerting_configured()

    def test_a_blank_webhook_secret_counts_as_unset(self) -> None:
        """An unset GitHub Actions secret expands to the empty string, not to
        nothing -- so `ALERT_DISCORD_WEBHOOK_URL=` reaches the process as `""`
        and would otherwise be posted to as a URL."""
        assert not Settings(_env_file=None, alert_discord_webhook_url="").alerting_configured()
        assert not Settings(_env_file=None, alert_discord_webhook_url="   ").alerting_configured()

    def test_the_pattern_confidence_floor_defers_to_the_measured_one(self) -> None:
        """Unset here, so the number lives in one place -- next to the
        distribution it was measured from."""
        assert Settings(_env_file=None).alert_pattern_min_confidence is None

    def test_the_pattern_confidence_floor_can_be_overridden(self) -> None:
        assert (
            Settings(_env_file=None, alert_pattern_min_confidence=85.0).alert_pattern_min_confidence
            == 85.0
        )

    def test_config_does_not_drag_the_analysis_stack_into_every_import(self) -> None:
        """`config` is imported by both front ends, the API and every script.

        Importing `alerting.rules` from it to read one float would pull in
        `analysis.scoring` and the indicator library behind it, and would become
        a circular import the first time scoring needs a setting.
        """
        import quantpulse.config as config_module

        source = Path(config_module.__file__).read_text()
        assert "from quantpulse" not in source and "import quantpulse" not in source


class TestPaperTradingGate:
    """Whether the forward test may place an order, and with whose account.

    Two credentials, both repo secrets, both unset by default -- every fork of
    this public repo runs the same refresh workflow, and a fork must neither
    fail nor trade because of it. There is deliberately no base-URL setting
    here: see `execution.alpaca` for why the endpoint is not configurable.
    """

    def test_it_is_off_until_both_credentials_are_present(self) -> None:
        assert not Settings(_env_file=None).paper_trading_configured()
        assert not Settings(_env_file=None, alpaca_api_key_id="PK123").paper_trading_configured()
        assert not Settings(_env_file=None, alpaca_api_secret_key="shh").paper_trading_configured()

    def test_both_credentials_turn_it_on(self) -> None:
        assert Settings(
            _env_file=None, alpaca_api_key_id="PK123", alpaca_api_secret_key="shh"
        ).paper_trading_configured()

    def test_blank_secrets_count_as_unset(self) -> None:
        assert not Settings(
            _env_file=None, alpaca_api_key_id="  ", alpaca_api_secret_key="shh"
        ).paper_trading_configured()

    def test_the_kill_switch_works_without_unsetting_the_credentials(self) -> None:
        assert not Settings(
            _env_file=None,
            alpaca_api_key_id="PK123",
            alpaca_api_secret_key="shh",
            paper_trading_enabled=False,
        ).paper_trading_configured()

    def test_there_is_no_setting_that_can_point_it_at_live_trading(self) -> None:
        """The guard in `execution.alpaca` is only as good as the absence of a
        knob beside it. A `base_url`/`endpoint` setting added here later would
        re-open exactly the hole that module is built to close.
        """
        fields = set(Settings.model_fields)
        for name in fields:
            assert "alpaca" not in name or not name.endswith(("url", "host", "endpoint")), (
                f"{name} looks like a configurable Alpaca endpoint; the paper host must "
                f"stay a module constant in execution.alpaca"
            )
