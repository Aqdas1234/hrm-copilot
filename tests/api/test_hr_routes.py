"""Login, tokens and role-protected HR endpoints, against a seeded in-memory database."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from hrm.api.app import create_app
from hrm.api.hr_routes import AuthConfig
from hrm.auth.context import UserContext
from hrm.auth.tokens import create_token
from hrm.db.models import AuditLog, User
from hrm.workflows.tickets import TicketService
from tests.api.helpers import PASSWORD, SECRET, employee_id, login
from tests.api.test_app import FakeService


def actions(factory: sessionmaker[Session]) -> list[str]:
    with factory() as s:
        return [
            a for a in s.scalars(select(AuditLog.action).order_by(AuditLog.id)) if a != "db.seed"
        ]


# --- login --------------------------------------------------------------------------------


def test_login_returns_a_bearer_token(client: TestClient) -> None:
    body = client.post(
        "/auth/login", json={"email": "alice.park@example.com", "password": PASSWORD}
    ).json()

    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 3600
    claims = jwt.decode(body["access_token"], SECRET, algorithms=["HS256"], issuer="hrm-copilot")
    assert claims["role"] == "employee"


def test_wrong_password_and_unknown_email_give_the_same_401(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    wrong = client.post("/auth/login", json={"email": "alice.park@example.com", "password": "nope"})
    unknown = client.post("/auth/login", json={"email": "nobody@example.com", "password": "nope"})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # cannot tell which emails exist
    assert actions(factory) == ["auth.login_failed", "auth.login_failed"]


def test_generated_employees_cannot_log_in(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as s:
        email = s.scalar(
            select(User.email).where(
                User.email.not_like("%.park@%"),
                User.email.not_like("%.chen@%"),
                User.email.not_like("%.khan@%"),
                User.email.not_like("%.reyes@%"),
            )
        )
    assert (
        client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 401
    )


@pytest.mark.parametrize("secret", [None, "too-short"])
def test_login_without_a_usable_secret_is_503(
    factory: sessionmaker[Session], secret: str | None
) -> None:
    app = create_app(
        FakeService(), session_factory=factory, auth=AuthConfig(secret, ttl_minutes=60)
    )
    with TestClient(app) as client:
        response = client.post(
            "/auth/login", json={"email": "alice.park@example.com", "password": PASSWORD}
        )

    assert response.status_code == 503


# --- tokens -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        {},  # no token
        {"Authorization": "Bearer not-a-jwt"},
        {
            "Authorization": "Bearer "
            + jwt.encode(
                {"sub": "1", "iss": "hrm-copilot", "iat": 0, "exp": 9e9},
                "wrong-secret-xxxxxxxxxxxxxxxxxxxxxxxx",
                algorithm="HS256",
            )
        },
    ],
    ids=["missing", "malformed", "forged with another secret"],
)
def test_requests_without_a_valid_token_are_401(client: TestClient, header: dict[str, str]) -> None:
    response = client.get("/me", headers=header)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_tampered_token_is_rejected(client: TestClient, factory: sessionmaker[Session]) -> None:
    token = login(client, "alice.park")["Authorization"].removeprefix("Bearer ")
    header, _original_payload, signature = token.split(".")
    claims = jwt.decode(token, options={"verify_signature": False})
    claims["sub"] = str(employee_id(factory, "carol.reyes"))  # try to become someone else
    forged_payload = jwt.encode(claims, "x" * 40, algorithm="HS256").split(".")[1]

    response = client.get(
        "/me", headers={"Authorization": f"Bearer {header}.{forged_payload}.{signature}"}
    )

    assert response.status_code == 401


def test_expired_token_is_rejected(client: TestClient, factory: sessionmaker[Session]) -> None:
    with factory() as s:
        user = s.scalar(select(User).where(User.email == "alice.park@example.com"))
        assert user is not None
        token = create_token(
            user, secret=SECRET, ttl_minutes=60, now=datetime.now(UTC) - timedelta(hours=2)
        )

    assert client.get("/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_disabled_account_is_locked_out_immediately(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    headers = login(client, "alice.park")
    with factory() as s:
        user = s.scalar(select(User).where(User.email == "alice.park@example.com"))
        assert user is not None
        user.is_active = False
        s.commit()

    assert client.get("/me", headers=headers).status_code == 401  # token still valid, account not


# --- role-protected endpoints ------------------------------------------------------------


def test_employee_sees_own_data(client: TestClient) -> None:
    headers = login(client, "alice.park")

    assert client.get("/me", headers=headers).json()["name"] == "Alice Park"
    assert len(client.get("/me/leave", headers=headers).json()) == 2
    assert client.get("/me/salary", headers=headers).json()["currency"] == "USD"
    assert client.get("/team", headers=headers).json() == []


def test_employee_is_refused_other_people_and_the_refusal_is_audited(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    headers = login(client, "alice.park")
    bob = employee_id(factory, "bob.chen")

    assert client.get(f"/employees/{bob}", headers=headers).status_code == 403
    assert client.get(f"/employees/{bob}/salary?reason=curious", headers=headers).status_code == 403
    assert actions(factory).count("access.denied") == 2  # committed despite the failed requests


def test_manager_sees_team_and_approvals_but_not_salaries(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    headers = login(client, "bob.chen")
    alice = employee_id(factory, "alice.park")

    assert {e["name"] for e in client.get("/team", headers=headers).json()} >= {
        "Alice Park",
        "Hira Khan",
    }
    assert len(client.get("/team/approvals", headers=headers).json()) == 2
    claims = client.get("/team/expense-approvals", headers=headers).json()
    assert [(c["description"], c["currency"]) for c in claims] == [("External monitor", "PKR")]
    assert client.get(f"/employees/{alice}", headers=headers).status_code == 200
    assert (
        client.get(f"/employees/{alice}/salary?reason=raise planning", headers=headers).status_code
        == 403
    )


def test_hr_admin_reads_any_salary_with_a_recorded_reason(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    headers = login(client, "carol.reyes")
    hira = employee_id(factory, "hira.khan")

    missing_reason = client.get(f"/employees/{hira}/salary", headers=headers)
    ok = client.get(f"/employees/{hira}/salary?reason=annual review", headers=headers)

    assert missing_reason.status_code == 422
    assert ok.status_code == 200 and ok.json()["currency"] == "PKR"
    with factory() as s:
        entry = s.scalar(select(AuditLog).where(AuditLog.action == "salary.read"))
        assert entry is not None and entry.details == {"reason": "annual review"}


def test_login_returns_the_role_for_the_ui(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"email": "carol.reyes@example.com", "password": PASSWORD}
    )
    assert response.json()["role"] == "hr_admin"


def test_users_see_only_their_own_tickets_and_hr_sees_the_queue(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as s:
        alice = s.scalar(select(User).where(User.email == "alice.park@example.com"))
        assert alice is not None
        TicketService(s, UserContext.from_user(alice)).open("payroll", "Tax code", "Details")
        s.commit()

    mine = client.get("/me/tickets", headers=login(client, "alice.park")).json()
    others = client.get("/me/tickets", headers=login(client, "bob.chen")).json()
    queue = client.get("/tickets/open", headers=login(client, "carol.reyes")).json()

    assert [t["subject"] for t in mine] == ["Tax code"]
    assert others == []
    assert [t["subject"] for t in queue] == ["Tax code"]
