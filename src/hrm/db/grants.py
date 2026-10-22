"""Database rights for the read-only analytics account (MySQL).

The account may read only the analytics views. The views run with their definer's rights
(SQL SECURITY DEFINER), so it needs nothing on the tables behind them: even SQL that got
past the checker cannot read `compensation` or `users`.

Granting needs an admin connection, which only the one-off set-up job has; the API never
does. The view list is the one the SQL checker uses, so grants and checks cannot drift.
"""

from sqlalchemy import Engine, text

from hrm.analytics.schema import VIEW_NAMES
from hrm.core.errors import ConfigurationError

ANALYTICS_ACCOUNT = "hrm_analytics"


def analytics_grants(database: str, account: str = ANALYTICS_ACCOUNT) -> list[str]:
    return [
        f"GRANT SELECT ON `{database}`.`{view}` TO '{account}'@'%'" for view in sorted(VIEW_NAMES)
    ]


def grant_analytics_access(admin: Engine, database: str) -> list[str]:
    """Grant SELECT on each analytics view; safe to repeat. Returns the statements run."""
    if admin.dialect.name != "mysql":
        raise ConfigurationError("Analytics grants are for MySQL; SQLite has no accounts.")
    statements = analytics_grants(database)
    with admin.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
    return statements
