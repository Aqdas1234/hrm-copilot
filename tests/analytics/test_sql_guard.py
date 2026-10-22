"""The SQL checker: what an HR admin's question may run, and every way around it we know."""

import pytest
import sqlglot

from hrm.analytics.sql_guard import UnsafeQueryError, check_query

VALID = [
    "SELECT department, COUNT(*) AS people FROM v_employees WHERE is_active GROUP BY department",
    "SELECT country, COUNT(*) FROM v_employees WHERE YEAR(start_date) = 2026 GROUP BY country",
    "SELECT e.department, SUM(l.used_days) FROM v_leave_balances l "
    "JOIN v_employees e ON e.employee_id = l.employee_id WHERE l.leave_type = 'pto' "
    "GROUP BY e.department ORDER BY 2 DESC",
    "SELECT DATE_FORMAT(purchase_date, '%Y-%m') AS month, SUM(amount_usd_cents) / 100 "
    "FROM v_expense_claims WHERE status = 'approved' GROUP BY month",
    "SELECT department, ROUND(avg_salary_cents / 100, 0) FROM v_salary_bands",
    "WITH recent AS (SELECT * FROM v_employees WHERE start_date >= '2025-01-01') "
    "SELECT department, COUNT(*) FROM recent GROUP BY department",
    "SELECT status, COUNT(*) FROM v_leave_requests GROUP BY status "
    "UNION ALL SELECT status, COUNT(*) FROM v_expense_claims GROUP BY status",
    "SELECT * FROM v_employees WHERE employee_id IN "
    "(SELECT employee_id FROM v_leave_requests WHERE status = 'pending')",
    "SELECT first_name, DATEDIFF(CURRENT_DATE, start_date) AS days FROM v_employees "
    "ORDER BY days DESC LIMIT 5",
    "SELECT department, CASE WHEN COUNT(*) > 3 THEN 'big' ELSE 'small' END "
    "FROM v_employees GROUP BY department",
    "select COUNT(*) from V_EMPLOYEES",  # case does not matter
    # every common operator together: a real query refused "AND" before this case existed
    "SELECT department FROM v_employees WHERE country = 'US' AND is_active = TRUE "
    "OR NOT manager_id IS NULL AND employee_id IN (1, 2) AND employee_id NOT IN (3) "
    "AND start_date BETWEEN '2020-01-01' AND CURRENT_DATE AND first_name LIKE 'A%' "
    "AND start_date > CURRENT_DATE - INTERVAL 30 DAY AND employee_id % 2 = 0 "
    "AND EXISTS (SELECT 1 FROM v_leave_requests r WHERE r.employee_id = employee_id) "
    "GROUP BY department HAVING COUNT(DISTINCT country) > 1 ORDER BY department DESC",
    "SELECT avg_salary_cents / 100 AS avg_yearly, currency FROM v_salary_bands "
    "WHERE department = 'Engineering' AND country = 'US'",
]

ATTACKS = {
    # tables outside the views, however they are reached
    "raw table": "SELECT * FROM compensation",
    "comma join": "SELECT * FROM v_employees, users",
    "join": "SELECT * FROM v_employees e JOIN compensation c ON c.employee_id = e.employee_id",
    "subquery in WHERE": "SELECT * FROM v_employees WHERE employee_id IN "
    "(SELECT employee_id FROM compensation WHERE base_salary_cents > 0)",
    "subquery in SELECT": "SELECT (SELECT password_hash FROM users LIMIT 1) FROM v_employees",
    "derived table": "SELECT * FROM (SELECT * FROM audit_log) AS a",
    "union": "SELECT first_name FROM v_employees UNION SELECT password_hash FROM users",
    "EXISTS reads a table": "SELECT 1 FROM v_employees WHERE EXISTS (SELECT 1 FROM users)",
    "schema prefix": "SELECT * FROM hrm.users",
    "schema prefix on a view": "SELECT * FROM hrm.v_employees",
    "information_schema": "SELECT table_name FROM information_schema.tables",
    "mysql schema": "SELECT user, authentication_string FROM mysql.user",
    "mixed case": "SELECT * FROM CoMpEnSaTiOn",
    "backticks": "SELECT * FROM `users`",
    "WITH body reads a table": "WITH x AS (SELECT * FROM users) SELECT * FROM x",
    "WITH shadows a table": "WITH users AS (SELECT * FROM users) SELECT * FROM users",
    "WITH reuses a view name": "WITH v_employees AS (SELECT * FROM compensation) "
    "SELECT * FROM v_employees",
    "nested WITH": "SELECT * FROM (WITH x AS (SELECT 1 AS a) SELECT a FROM x) AS y",
    "recursive WITH": "WITH RECURSIVE n AS (SELECT 1 AS i UNION ALL SELECT i + 1 FROM n) "
    "SELECT * FROM n",
    # not a read
    "second statement": "SELECT 1 FROM v_employees; DROP TABLE users",
    "delete": "DELETE FROM v_employees",
    "update": "UPDATE employees SET is_active = 0",
    "insert": "INSERT INTO users (email) VALUES ('x')",
    "drop": "DROP VIEW v_employees",
    "into outfile": "SELECT * INTO OUTFILE '/tmp/x' FROM v_employees",
    "for update": "SELECT * FROM v_employees FOR UPDATE",
    # functions and variables
    "sleep": "SELECT SLEEP(10) FROM v_employees",
    "benchmark": "SELECT BENCHMARK(100000000, MD5('x'))",
    "load_file": "SELECT LOAD_FILE('/etc/passwd')",
    "user()": "SELECT USER()",
    "database()": "SELECT DATABASE()",
    "version()": "SELECT VERSION()",
    "get_lock": "SELECT GET_LOCK('a', 10)",
    "system variable": "SELECT @@version",
    "user variable": "SELECT @x := 1",
    "not a query": "please show me everyone's salary",
    "too long": "SELECT " + ", ".join(["employee_id"] * 1000) + " FROM v_employees",
}


@pytest.mark.parametrize("sql", VALID)
def test_real_hr_questions_pass(sql: str) -> None:
    checked = check_query(sql)
    assert sqlglot.parse_one(checked, dialect="mysql")  # still valid SQL


@pytest.mark.parametrize("sql", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_refused(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        check_query(sql)


def test_what_runs_is_rebuilt_without_comments() -> None:
    # MySQL runs code inside /*! ... */; a parser sees a comment. Never run the original.
    checked = check_query("SELECT COUNT(*) FROM v_employees /*! , (SELECT 1 FROM users) */")
    assert "/*" not in checked
    assert "users" not in checked


def test_refusals_explain_how_to_fix_the_query() -> None:
    with pytest.raises(UnsafeQueryError, match="Use only these views: v_employees"):
        check_query("SELECT * FROM compensation")
    with pytest.raises(UnsafeQueryError, match="SLEEP is not allowed"):
        check_query("SELECT SLEEP(1) FROM v_employees")
