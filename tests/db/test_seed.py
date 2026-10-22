from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from hrm.auth.passwords import verify_password
from hrm.db.models import (
    Compensation,
    Employee,
    ExpenseClaim,
    LeaveRequest,
    LeaveType,
    PublicHoliday,
    RequestStatus,
    Role,
    User,
)
from hrm.db.seed import DatabaseNotEmptyError, seed
from hrm.db.session import create_schema, make_engine

TODAY = date(2026, 10, 22)  # a Thursday
PASSWORD = "demo-pass"


@pytest.fixture
def seeded(session: Session) -> Session:
    seed(session, today=TODAY, demo_password=PASSWORD)
    return session


def employee(s: Session, email: str) -> Employee:
    found = s.scalar(select(Employee).where(Employee.email == email))
    assert found is not None
    return found


def role_of(s: Session, email: str) -> Role:
    found = s.scalar(select(User.role).where(User.email == email))
    assert found is not None
    return found


def test_counts(seeded: Session) -> None:
    assert seeded.scalar(select(func.count()).select_from(Employee)) == 30
    assert seeded.scalar(select(func.count()).select_from(User)) == 30
    assert seeded.scalar(select(func.count()).select_from(Compensation)) == 30


def test_scripted_people_roles_and_reporting_lines(seeded: Session) -> None:
    alice = employee(seeded, "alice.park@example.com")
    hira = employee(seeded, "hira.khan@example.com")
    bob = employee(seeded, "bob.chen@example.com")

    assert alice.manager is bob and hira.manager is bob
    assert hira.country == "PK"
    assert role_of(seeded, "carol.reyes@example.com") is Role.HR_ADMIN
    assert role_of(seeded, "bob.chen@example.com") is Role.MANAGER
    assert role_of(seeded, "alice.park@example.com") is Role.EMPLOYEE


def test_generated_managers_get_the_manager_role(seeded: Session) -> None:
    managers = seeded.scalars(select(Employee).where(Employee.reports.any())).all()

    assert len(managers) >= 4
    for m in managers:
        assert role_of(seeded, m.email) in {Role.MANAGER, Role.HR_ADMIN}


def test_only_demo_users_can_log_in_with_the_demo_password(seeded: Session) -> None:
    hashes = dict(seeded.execute(select(User.email, User.password_hash)).all())
    can_log_in = {email for email, h in hashes.items() if verify_password(PASSWORD, h)}

    assert can_log_in == {
        "carol.reyes@example.com",
        "bob.chen@example.com",
        "alice.park@example.com",
        "hira.khan@example.com",
    }


def test_emails_are_unique_and_use_the_reserved_example_domain(seeded: Session) -> None:
    emails = seeded.scalars(select(Employee.email)).all()

    assert len(set(emails)) == len(emails)
    assert all(e.endswith("@example.com") for e in emails)


def test_salaries_are_in_local_currency_and_plausible(seeded: Session) -> None:
    rows = seeded.execute(
        select(Employee.country, Compensation.currency, Compensation.base_salary_cents).join(
            Compensation, Compensation.employee_id == Employee.id
        )
    ).all()

    for country, currency, cents in rows:
        assert currency == {"US": "USD", "GB": "GBP", "DE": "EUR", "PK": "PKR"}[country]
        low = {"US": 110_000, "GB": 70_000, "DE": 70_000, "PK": 3_000_000}[country]
        assert cents >= low * 100


def test_policy_based_data(seeded: Session) -> None:
    annual = dict(seeded.execute(select(LeaveType.code, LeaveType.annual_days)).all())
    laptop = seeded.scalar(
        select(ExpenseClaim).where(ExpenseClaim.description == "Replacement laptop")
    )
    stand = seeded.scalar(select(ExpenseClaim).where(ExpenseClaim.description == "Laptop stand"))

    assert annual["pto"] is None  # flexible PTO: no set limit
    assert annual["parental"] == 80
    assert laptop is not None and laptop.is_company_property  # over $1,000
    assert stand is not None and not stand.is_company_property


def test_leave_requests_are_on_weekdays_and_pending_ones_go_to_the_manager(seeded: Session) -> None:
    requests = seeded.scalars(select(LeaveRequest)).all()
    alice = employee(seeded, "alice.park@example.com")

    assert all(r.start_date.weekday() < 5 and r.end_date.weekday() < 5 for r in requests)
    pending = [
        r for r in requests if r.status is RequestStatus.PENDING and r.employee_id == alice.id
    ]
    assert pending and pending[0].approver_id == alice.manager_id


def test_pakistan_holidays_are_stored_for_this_and_next_year(seeded: Session) -> None:
    days = seeded.scalars(select(PublicHoliday.day).where(PublicHoliday.country == "PK")).all()

    assert date(2026, 8, 14) in days and date(2027, 3, 23) in days
    assert len(days) == 12


def test_seeding_is_deterministic() -> None:
    def snapshot() -> list[tuple[str, str, int]]:
        engine = make_engine("sqlite://")
        create_schema(engine)
        with sessionmaker(engine)() as s:
            seed(s, today=TODAY, demo_password=PASSWORD)
            rows = s.execute(
                select(Employee.email, Employee.job_title, Compensation.base_salary_cents)
                .join(Compensation, Compensation.employee_id == Employee.id)
                .order_by(Employee.employee_code)
            ).all()
            return [(email, title, cents) for email, title, cents in rows]

    assert snapshot() == snapshot()


def test_refuses_to_seed_a_database_that_has_data(seeded: Session) -> None:
    with pytest.raises(DatabaseNotEmptyError):
        seed(seeded, today=TODAY, demo_password=PASSWORD)
