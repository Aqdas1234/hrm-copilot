"""Security suite: who may call what, no salary leaks, no internals in error responses.

ACCESS_RULES lists every endpoint the API exposes; test_every_endpoint_has_an_access_rule
fails when a new endpoint is added without deciding who may use it.
"""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from hrm.api.app import create_app
from hrm.api.hr_routes import AuthConfig
from hrm.core.errors import LLMError
from hrm.db.models import Compensation
from tests.api.helpers import DEMO_PEOPLE, SECRET, employee_id, login
from tests.api.test_app import FakeService

PUBLIC = "public"
AUTHENTICATED = "authenticated: result depends on the caller's role"
# Conversations are keyed by the logged-in user; tested in test_chat.py.
OWN_CONVERSATION = "authenticated: only the caller's own conversations"

# Every endpoint, and who may call it. Role-level detail is in MATRIX below.
ACCESS_RULES = {
    ("GET", "/health"): PUBLIC,
    ("POST", "/auth/login"): PUBLIC,
    # Policy documents are company-wide information, not personal data.
    ("POST", "/rag/search"): PUBLIC,
    ("POST", "/rag/ask"): PUBLIC,
    ("GET", "/me"): AUTHENTICATED,
    ("GET", "/me/leave"): AUTHENTICATED,
    ("GET", "/me/salary"): AUTHENTICATED,
    ("GET", "/team"): AUTHENTICATED,
    ("GET", "/team/approvals"): AUTHENTICATED,
    ("GET", "/team/expense-approvals"): AUTHENTICATED,
    ("GET", "/me/tickets"): AUTHENTICATED,
    ("GET", "/tickets/open"): AUTHENTICATED,
    ("GET", "/employees"): AUTHENTICATED,
    ("GET", "/employees/{employee_id}"): AUTHENTICATED,
    ("GET", "/employees/{employee_id}/salary"): AUTHENTICATED,
    ("POST", "/chat/{conversation_id}/messages"): OWN_CONVERSATION,
    ("POST", "/chat/{conversation_id}/confirm"): OWN_CONVERSATION,
    ("GET", "/chat/{conversation_id}"): OWN_CONVERSATION,
    ("GET", "/chat"): OWN_CONVERSATION,
    ("DELETE", "/chat/{conversation_id}"): OWN_CONVERSATION,
}

ACTORS = ["anonymous", "alice", "bob", "carol"]
# (path with {person} placeholders, expected status for anonymous, alice, bob, carol)
# Alice and Hira report to Bob; Carol is HR admin.
MATRIX: list[tuple[str, tuple[int, int, int, int]]] = [
    ("/me", (401, 200, 200, 200)),
    ("/me/leave", (401, 200, 200, 200)),
    ("/me/salary", (401, 200, 200, 200)),
    ("/team", (401, 200, 200, 200)),
    ("/team/approvals", (401, 200, 200, 200)),
    ("/team/expense-approvals", (401, 200, 200, 200)),  # only claims assigned to the caller
    ("/me/tickets", (401, 200, 200, 200)),
    ("/tickets/open", (401, 403, 403, 200)),  # HR admins only
    ("/employees", (401, 403, 403, 200)),  # HR admins only
    ("/employees/{alice}", (401, 200, 200, 200)),
    ("/employees/{hira}", (401, 403, 200, 200)),
    ("/employees/{bob}", (401, 403, 200, 200)),
    ("/employees/{carol}", (401, 403, 403, 200)),
    ("/employees/999999", (401, 403, 403, 403)),  # unknown id looks the same as forbidden
    ("/employees/{alice}/salary?reason=check", (401, 200, 403, 200)),  # manager: never pay
    ("/employees/{hira}/salary?reason=check", (401, 403, 403, 200)),
    ("/employees/{bob}/salary?reason=check", (401, 403, 200, 200)),
    ("/employees/{carol}/salary?reason=check", (401, 403, 403, 200)),
]

SALARY_PATHS = ("/salary",)
SALARY_KEYS = ("salary", "compensation", "yearly_amount", "base_salary")


def resolve(path: str, factory: sessionmaker[Session]) -> str:
    for short, name in DEMO_PEOPLE.items():
        path = path.replace(f"{{{short}}}", str(employee_id(factory, name)))
    return path


def headers_for(client: TestClient, actor: str) -> dict[str, str]:
    return {} if actor == "anonymous" else login(client, DEMO_PEOPLE[actor])


# --- 1. every endpoint has a rule, and the matrix holds for every role -------------------


def test_every_endpoint_has_an_access_rule(client: TestClient) -> None:
    exposed = {
        (method.upper(), path)
        for path, operations in client.app.openapi()["paths"].items()  # type: ignore[attr-defined]
        for method in operations
    }

    assert exposed == set(ACCESS_RULES), (
        "Add new endpoints to ACCESS_RULES (and MATRIX if protected)"
    )


def test_matrix_covers_every_protected_endpoint() -> None:
    protected = {path for (_, path), rule in ACCESS_RULES.items() if rule == AUTHENTICATED}
    covered = {
        "/employees/{employee_id}" + ("/salary" if "/salary" in p else "")
        if p.startswith("/employees/")
        else p.split("?")[0]
        for p, _ in MATRIX
    }

    assert protected == covered


@pytest.mark.parametrize(("path", "expected"), MATRIX, ids=[p for p, _ in MATRIX])
def test_access_matrix(
    client: TestClient, factory: sessionmaker[Session], path: str, expected: tuple[int, ...]
) -> None:
    url = resolve(path, factory)
    actual = tuple(
        client.get(url, headers=headers_for(client, actor)).status_code for actor in ACTORS
    )

    assert dict(zip(ACTORS, actual, strict=True)) == dict(zip(ACTORS, expected, strict=True))


# --- 2. salary never appears outside the salary endpoints ---------------------------------


def keys_in(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_in(v)}
    if isinstance(value, list):
        return {k for item in value for k in keys_in(item)}
    return set()


@pytest.mark.parametrize("actor", ["alice", "bob", "carol"])
def test_no_salary_in_non_salary_responses(
    client: TestClient, factory: sessionmaker[Session], actor: str
) -> None:
    with factory() as s:
        amounts = {c / 100 for c in s.scalars(select(Compensation.base_salary_cents))}
    headers = headers_for(client, actor)
    paths = [resolve(p, factory) for p, _ in MATRIX if not any(x in p for x in SALARY_PATHS)]

    for url in paths:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            continue
        body = response.json()
        leaked_keys = {k for k in keys_in(body) if any(s in k.lower() for s in SALARY_KEYS)}
        assert not leaked_keys, f"{actor} {url} leaked {leaked_keys}"
        text = json.dumps(body)
        assert not [a for a in amounts if f"{a:.1f}" in text or f"{int(a)}" in text.split()], url


# --- 3. error responses never reveal internals ----------------------------------------------

INTERNALS = (
    "Traceback",
    'File "',
    ".py",
    "sqlalchemy",
    "SELECT ",
    "pymysql",
    "/Users/",
    "password_hash",
    SECRET,
    "scrypt$",
)


def assert_no_internals(text: str) -> None:
    found = [marker for marker in INTERNALS if marker in text]
    assert not found, f"response leaks {found}: {text[:200]}"


def test_error_responses_do_not_reveal_internals(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    alice = login(client, "alice.park")
    responses = [
        client.get("/me"),  # 401
        client.get("/me", headers={"Authorization": "Bearer x.y.z"}),  # 401
        client.get(resolve("/employees/{carol}", factory), headers=alice),  # 403
        client.get("/employees/not-a-number", headers=alice),  # 422
        client.post(
            "/auth/login", json={"email": "alice.park@example.com", "password": "wrong"}
        ),  # 401
        client.post("/auth/login", json={"email": "x"}),  # 422
        client.get("/no-such-endpoint"),  # 404
    ]

    for response in responses:
        assert response.status_code >= 400
        assert_no_internals(response.text)


def test_llm_failure_details_stay_in_the_logs(factory: sessionmaker[Session]) -> None:
    failing = FakeService(error=LLMError('Groq 500 at File "/Users/dev/app.py", SELECT * FROM x'))
    app = create_app(failing, session_factory=factory, auth=AuthConfig(SECRET, ttl_minutes=60))
    with TestClient(app) as client:
        response = client.post("/rag/ask", json={"question": "q"})

    assert response.status_code == 502
    assert_no_internals(response.text)


def test_validation_errors_do_not_echo_the_submitted_password(client: TestClient) -> None:
    secret_password = "MyRealPassword-" + "x" * 200  # too long: fails validation

    response = client.post(
        "/auth/login", json={"email": "alice.park@example.com", "password": secret_password}
    )

    assert response.status_code == 422
    assert "MyRealPassword" not in response.text
