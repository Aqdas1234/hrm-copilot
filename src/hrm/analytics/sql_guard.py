"""Decide whether model-written SQL may run. Deny by default; parsed, never pattern-matched.

The query is parsed with sqlglot (MySQL dialect) and every part of the tree is checked:
- exactly one statement, and it reads (SELECT, possibly with WITH / UNION);
- every table anywhere in it (subqueries and WITH bodies included) is an analytics view,
  named without a schema ("hrm.users" and "information_schema.x" are refused);
- WITH only at the start, not recursive, and its names never match a real table: in
  `WITH users AS (SELECT * FROM users)` the inner `users` is the real table;
- every function is on an allow-list (aggregates, dates, text, arithmetic helpers);
  anything unknown (SLEEP, BENCHMARK, LOAD_FILE, USER(), ...) is refused;
- no variables (@x, @@version), no locking (FOR UPDATE), no SELECT ... INTO.

The caller runs the SQL this returns, rebuilt from the checked tree without comments,
never the model's original text: MySQL executes code hidden in /*! ... */ comments, which
a parser treats as a comment. What runs is exactly what was checked.

The database account that runs it can read only the views (see analytics/runner.py), so
this check is the second line of defence, not the only one.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from hrm.analytics.schema import VIEW_NAMES
from hrm.core.errors import HRMError
from hrm.db.models import Base

MAX_SQL_CHARS = 4_000

ALLOWED_FUNCTIONS = frozenset(
    {
        # logical operators that sqlglot models as functions (EXISTS bodies are still checked)
        "And", "Or", "Xor", "Exists",
        # aggregates and window functions
        "Count", "Sum", "Avg", "Min", "Max", "RowNumber", "Rank",
        # conditionals and casts
        "Case", "If", "Coalesce", "Nullif", "Cast", "Greatest", "Least",
        # numbers
        "Round", "Abs", "Floor", "Ceil",
        # dates
        "Year", "Month", "CurrentDate", "DateDiff", "DateSub", "Extract", "TimestampDiff",
        "TimeToStr", "TsOrDsToTimestamp", "TsOrDsToDate",
        # text
        "Lower", "Upper", "Concat", "Length", "Trim", "Substring",
    }
)  # fmt: skip
_WRITES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Into)
_VARIABLES = (exp.Parameter, exp.SessionParameter, exp.Placeholder, exp.PropertyEQ)
# Names a WITH clause may not take: they would hide which object a name refers to.
_REAL_TABLES = frozenset({*Base.metadata.tables, "alembic_version"})


class UnsafeQueryError(HRMError):
    """The SQL may not run. The message says why, so the model can write a valid query."""


def check_query(sql: str, allowed: frozenset[str] = VIEW_NAMES) -> str:
    """Return the SQL to run (rebuilt from the checked tree), or raise UnsafeQueryError."""
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeQueryError("The query is too long.")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="mysql") if s is not None]
    except SqlglotError as exc:  # parse and tokenizer errors alike: never let one escape
        raise UnsafeQueryError(f"The query is not valid MySQL: {_first_line(exc)}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError("Write exactly one SELECT statement.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise UnsafeQueryError("Only SELECT queries are allowed.")

    _check_nodes(tree)
    _check_tables(tree, allowed)
    return tree.sql(dialect="mysql", comments=False)


def _check_nodes(tree: exp.Expression) -> None:
    for node in tree.walk():
        if isinstance(node, _WRITES):
            raise UnsafeQueryError("Only SELECT queries are allowed.")
        if isinstance(node, _VARIABLES):
            raise UnsafeQueryError("Variables are not allowed.")
        if isinstance(node, exp.Lock):
            raise UnsafeQueryError("Locking clauses are not allowed.")
        if isinstance(node, exp.Func) and type(node).__name__ not in ALLOWED_FUNCTIONS:
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            raise UnsafeQueryError(f"The function {name.upper()} is not allowed.")


def _check_tables(tree: exp.Expression, allowed: frozenset[str]) -> None:
    ctes = _top_level_ctes(tree)
    if ctes & (allowed | _REAL_TABLES):
        raise UnsafeQueryError("WITH names must not reuse table or view names.")
    for table in tree.find_all(exp.Table):
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeQueryError("Use view names without a schema prefix.")
        name = table.name.lower()
        if name not in allowed and name not in ctes:
            views = ", ".join(sorted(allowed))
            raise UnsafeQueryError(f"Unknown table {name!r}. Use only these views: {views}.")


def _top_level_ctes(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for node in tree.find_all(exp.With):
        if node.parent is not tree:
            raise UnsafeQueryError("WITH is only allowed at the start of the query.")
        if node.args.get("recursive"):
            raise UnsafeQueryError("Recursive WITH is not allowed.")
        names |= {cte.alias_or_name.lower() for cte in node.expressions}
    return names


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0][:200]
