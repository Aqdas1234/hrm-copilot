"""Shared helpers for API tests that log in against the seeded database."""

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from hrm.db.models import User

PASSWORD = "demo-pass"
SECRET = "test-secret-at-least-32-bytes-long!!"
DEMO_PEOPLE = {
    "alice": "alice.park",
    "bob": "bob.chen",
    "carol": "carol.reyes",
    "hira": "hira.khan",
}


def login(client: TestClient, name: str, password: str = PASSWORD) -> dict[str, str]:
    response = client.post(
        "/auth/login", json={"email": f"{name}@example.com", "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def employee_id(factory: sessionmaker[Session], name: str) -> int:
    with factory() as s:
        user = s.scalar(select(User).where(User.email == f"{name}@example.com"))
        assert user is not None
        return user.employee_id
