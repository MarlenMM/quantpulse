from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantpulse.config import get_settings

engine = create_engine(get_settings().database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class SchemaOutOfDateError(RuntimeError):
    """The database is behind the migration head, so writes will fail mid-run."""


def assert_schema_current() -> None:
    """Fail immediately if the database is not migrated up to the latest revision.

    Called at the top of the long-running jobs. Without it, a missed
    `alembic upgrade head` surfaces as an `OperationalError: table X has no
    column named Y` from whichever step happens to write that column first --
    which is both hours into the run and several frames away from the actual
    cause. That is not hypothetical: a nightly refresh spent nine minutes
    fetching 503 tickers before its composite-scoring step died on a missing
    `fundamental_raw`, leaving the ratings table empty while prices, options and
    the regime row all landed and the job reported "partial".

    Cheap enough to run unconditionally (one query plus a directory scan), and
    the failure it replaces is expensive.
    """
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    heads = set(script.get_heads())

    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_heads()

    if set(current) != heads:
        raise SchemaOutOfDateError(
            f"database is at revision {sorted(current) or ['<none>']} but the code "
            f"expects {sorted(heads)}. Run `alembic upgrade head` before this job -- "
            "continuing would fail partway through, after hours of fetching."
        )
