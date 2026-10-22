"""Schema migrations (Alembic): `hrm db upgrade` brings any database to the latest schema.

create_all() only creates missing tables and never changes existing ones, so a deployed
database would silently fall behind the models. Every schema change is a versioned script
in migrations/versions; a test checks the migrations produce exactly the models' schema.

New migration after changing models.py:
    alembic revision --autogenerate -m "what changed"    (review the generated file!)
"""

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, Engine, inspect

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


def drop_everything(engine: Engine) -> None:
    """All tables, including Alembic's version table (for `hrm db seed --reset`)."""
    Base.metadata.drop_all(engine)
    if inspect(engine).has_table("alembic_version"):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE alembic_version")


def _config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def _is_unversioned_existing(connection: Connection) -> bool:
    tables = inspect(connection).get_table_names()
    return "alembic_version" not in tables and "employees" in tables


def _adopt(connection: Connection, config: Config) -> None:
    """A database created by create_all() before migrations existed: if it matches the
    baseline exactly, record it as the baseline version; never guess otherwise."""
    differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    if differences:
        raise SchemaMismatchError(
            f"The database has tables but no migration version, and differs from the "
            f"baseline schema ({len(differences)} differences). Back it up and reset it, or "
            f"stamp it by hand with `alembic stamp {BASELINE}` after checking."
        )
    command.stamp(config, BASELINE)
