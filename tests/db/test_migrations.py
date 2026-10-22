"""Migrations must build exactly the schema the models describe."""

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from hrm.db.migrate import (
    SchemaMismatchError,
    current_revision,
    drop_everything,
    head_revision,
    upgrade,
)
from hrm.db.models import Base
from hrm.db.session import create_schema, make_engine


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return make_engine(f"sqlite:///{tmp_path / 'hr.db'}")


def test_migrations_match_the_models(engine: Engine) -> None:
    upgrade(engine)
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    # Non-empty: models.py changed without a migration. Run
    #   alembic revision --autogenerate -m "..."  and review the generated file.
    assert differences == []


def test_upgrade_is_versioned_and_idempotent(engine: Engine) -> None:
    upgrade(engine)
    upgrade(engine)
    assert current_revision(engine) == head_revision()
    assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())


def test_check_constraints_are_created(engine: Engine) -> None:
    """compare_metadata does not compare CHECK constraints, so test one directly."""
    upgrade(engine)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text("INSERT INTO departments (name) VALUES ('X')"))
        connection.execute(
            text(
                "INSERT INTO exchange_rates (currency, day, usd_per_unit, published_on, source)"
                " VALUES ('EUR', '2026-10-22', -1, '2026-10-22', 'api')"
            )
        )


def test_reset_drops_the_version_table_too(engine: Engine) -> None:
    upgrade(engine)
    drop_everything(engine)
    assert inspect(engine).get_table_names() == []
    assert inspect(engine).get_view_names() == []

    upgrade(engine)  # a reset database is rebuilt, not mistaken for an up-to-date one
    assert "employees" in inspect(engine).get_table_names()


def test_a_database_created_before_migrations_is_adopted(engine: Engine) -> None:
    create_schema(engine)  # how `hrm db init` used to create it
    upgrade(engine)  # adopted as the baseline, then upgraded from there
    assert current_revision(engine) == head_revision()
    assert "v_employees" in inspect(engine).get_view_names()


def test_an_unversioned_database_that_differs_is_refused(engine: Engine) -> None:
    create_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE employees ADD COLUMN surprise INTEGER"))
    with pytest.raises(SchemaMismatchError):
        upgrade(engine)
    assert current_revision(engine) is None
