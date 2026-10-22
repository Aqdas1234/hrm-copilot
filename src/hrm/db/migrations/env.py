"""Alembic environment: runs migrations against the HR database.

Two entry points:
- `hrm db upgrade` (hrm.db.migrate) passes an open connection in config.attributes;
- the `alembic` command line (alembic.ini, for writing new migrations) falls back to the
  database URL from settings.
The URL is never written into the config text: Alembic's .ini handling treats "%" specially,
which would break passwords containing it.
"""

from alembic import context
from sqlalchemy import Connection

from hrm.core.config import get_settings
from hrm.db.models import Base
from hrm.db.session import make_engine

target_metadata = Base.metadata


def run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things; batch mode copies the table instead.
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    run(connection)
else:
    engine = make_engine(get_settings().database_url.get_secret_value())
    with engine.connect() as conn:
        run(conn)
    engine.dispose()
