"""Schema migrations (Alembic): `hrm db upgrade` brings any database to the latest schema.

create_all() only creates missing tables and never changes existing ones, so a deployed
database would silently fall behind the models. Every schema change is a versioned script
in migrations/versions; a test checks the migrations produce exactly the models' schema.

New migration after changing models.py:
    alembic revision --autogenerate -m "what changed"    (review the generated file!)
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, create_engine, inspect

from hrm.core.errors import HRMError
from hrm.db.models import Base

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
BASELINE = "0001"  # the schema databases had before migrations existed


class SchemaMismatchError(HRMError):
    """An unversioned database does not match the baseline schema; adopt it by hand."""


def upgrade(engine: Engine, revision: str = "head") -> None:
    with engine.begin() as connection:
        config = _config(connection)
        if _is_unversioned_existing(connection):
            _adopt(connection, config)
        command.upgrade(config, revision)


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def head_revision() -> str:
    """The latest migration in migrations/versions."""
    head = ScriptDirectory.from_config(_config(None)).get_current_head()
    assert head is not None
    return head


def drop_everything(engine: Engine) -> None:
    """Views, tables and Alembic's version table (for `hrm db seed --reset`).
    drop_all() knows only the models' tables, not the views migrations created."""
    with engine.begin() as connection:
        for view in inspect(connection).get_view_names():
            connection.exec_driver_sql(f"DROP VIEW {view}")
    Base.metadata.drop_all(engine)
    if inspect(engine).has_table("alembic_version"):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE alembic_version")


def _config(connection: Connection | None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def _is_unversioned_existing(connection: Connection) -> bool:
    tables = inspect(connection).get_table_names()
    return "alembic_version" not in tables and "employees" in tables


def _adopt(connection: Connection, config: Config) -> None:
    """A database created by create_all() before migrations existed: if its tables and
    columns are exactly the baseline's, record it as the baseline version (later
    migrations then apply normally); never guess otherwise.

    Compared with the baseline, not the current models: those include later migrations.
    """
    actual = _columns(connection)
    expected = _baseline_columns()
    if actual != expected:
        different = sorted(
            t for t in actual.keys() | expected.keys() if actual.get(t) != expected.get(t)
        )
        raise SchemaMismatchError(
            f"The database has tables but no migration version, and differs from the "
            f"baseline schema (tables: {', '.join(different)}). Back it up and reset it, or "
            f"stamp it by hand with `alembic stamp <revision>` after checking."
        )
    command.stamp(config, BASELINE)


def _baseline_columns() -> dict[str, set[str]]:
    scratch = create_engine("sqlite://")
    try:
        upgrade(scratch, BASELINE)
        with scratch.connect() as connection:
            return _columns(connection)
    finally:
        scratch.dispose()


def _columns(connection: Connection) -> dict[str, set[str]]:
    inspector = inspect(connection)
    return {
        table: {c["name"] for c in inspector.get_columns(table)}
        for table in inspector.get_table_names()
        if table != "alembic_version"
    }
