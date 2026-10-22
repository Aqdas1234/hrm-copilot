"""Engine and session creation. One engine per process; one session per unit of work."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from hrm.db.models import Base


def make_engine(url: str, *, echo: bool = False) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, echo=echo)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
            # SQLite ignores foreign keys unless asked; MySQL (InnoDB) always enforces them.
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        return engine
    # pool_pre_ping: replace connections MySQL closed while idle (wait_timeout).
    return create_engine(url, echo=echo, pool_pre_ping=True, pool_recycle=3600)


def create_schema(engine: Engine) -> None:
    """Create all tables that do not exist yet (no-op for existing ones)."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit on success, roll back on any error, always close."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
