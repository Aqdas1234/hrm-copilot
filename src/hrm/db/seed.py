"""Deterministic demo data for the HR database (`hrm db seed`).

Scripted people that tests and demos rely on:
  Carol Reyes  - HR admin (People, US): may see everything, including salaries
  Bob Chen     - Engineering manager (US): sees his reports, never their salaries
  Alice Park   - Engineer (US), reports to Bob
  Hira Khan    - Engineer (Pakistan), reports to Bob: exercises the holiday-API fallback
Plus generated colleagues (Faker with a fixed seed), so two runs give identical data.
Only the scripted users get the demo password; generated users cannot log in.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from faker import Faker
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hrm.auth.passwords import hash_password, unusable_password
from hrm.db.models import (
    AuditLog,
    Compensation,
    Department,
    Employee,
    EmploymentType,
    ExpenseClaim,
    HolidaySource,
    LeaveBalance,
    LeaveRequest,
    LeaveType,
    PublicHoliday,
    RequestStatus,
    Role,
    User,
)

EMAIL_DOMAIN = "example.com"  # reserved for documentation: never a real mailbox
DEPARTMENTS = ["Engineering", "Product", "Sales", "Customer Engineering", "Finance", "People"]
COUNTRIES = ["US", "US", "US", "GB", "DE", "PK"]  # weighted towards the US
# Yearly base salary ranges in each country's own currency (whole units, not cents).
SALARY_RANGE = {
    "US": (110_000, 190_000),
    "GB": (70_000, 120_000),
    "DE": (70_000, 120_000),
    "PK": (3_000_000, 9_000_000),
}
CURRENCY = {"US": "USD", "GB": "GBP", "DE": "EUR", "PK": "PKR"}
# Fixed demo rates to USD; real conversion comes from the currency API (B3.2).
DEMO_USD_RATE = {
    "USD": Decimal("1"),
    "GBP": Decimal("1.27"),
    "EUR": Decimal("1.08"),
    "PKR": Decimal("0.0036"),
}
# Pakistan's fixed-date public holidays: Nager.Date does not cover PK, so they live in the DB.
PK_HOLIDAYS = [
    ((2, 5), "Kashmir Solidarity Day"),
    ((3, 23), "Pakistan Day"),
    ((5, 1), "Labour Day"),
    ((8, 14), "Independence Day"),
    ((11, 9), "Iqbal Day"),
    ((12, 25), "Quaid-e-Azam Day"),
]


class DatabaseNotEmptyError(Exception):
    """Seeding would mix demo data with existing data; use reset instead."""


@dataclass(frozen=True)
class SeedSummary:
    employees: int
    users: int
    demo_users: list[str]
    leave_requests: int
    expense_claims: int
    holidays: int


def seed(session: Session, *, today: date, demo_password: str, seed_value: int = 42) -> SeedSummary:
    if session.scalar(select(func.count()).select_from(Employee)):
        raise DatabaseNotEmptyError("The database already has employees; reset it to re-seed.")
    fake = Faker("en_US")
    fake.seed_instance(seed_value)
    rng = random.Random(seed_value)
    demo_hash = hash_password(demo_password)

    departments = {name: Department(name=name) for name in DEPARTMENTS}
    session.add_all(departments.values())
    leave = {
        "pto": LeaveType(code="pto", name="Paid time off", annual_days=None),  # no set limit
        "parental": LeaveType(
            code="parental", name="Paid parental leave", annual_days=Decimal("80")
        ),
        "unpaid": LeaveType(code="unpaid", name="Unpaid leave", annual_days=None, paid=False),
    }
    session.add_all(leave.values())

    used_emails: set[str] = set()

    def unique_email(first: str, last: str) -> str:
        base = f"{first}.{last}".lower().replace(" ", "").replace("'", "")
        email, n = f"{base}@{EMAIL_DOMAIN}", 1
        while email in used_emails:  # Faker can repeat a name
            n += 1
            email = f"{base}{n}@{EMAIL_DOMAIN}"
        used_emails.add(email)
        return email

    def person(
        code: str,
        first: str,
        last: str,
        title: str,
        dept: str,
        country: str,
        manager: Employee | None,
        years: int,
    ) -> Employee:
        employee = Employee(
            employee_code=code,
            first_name=first,
            last_name=last,
            email=unique_email(first, last),
            job_title=title,
            department=departments[dept],
            manager=manager,
            country=country,
            employment_type=EmploymentType.FULL_TIME,
            start_date=today - timedelta(days=365 * years),
        )
        session.add(employee)
        return employee

    carol = person("E0001", "Carol", "Reyes", "Head of People", "People", "US", None, 6)
    bob = person("E0002", "Bob", "Chen", "Engineering Manager", "Engineering", "US", None, 5)
    alice = person("E0003", "Alice", "Park", "Software Engineer", "Engineering", "US", bob, 2)
    hira = person("E0004", "Hira", "Khan", "Software Engineer", "Engineering", "PK", bob, 1)
    scripted = {carol: Role.HR_ADMIN, bob: Role.MANAGER, alice: Role.EMPLOYEE, hira: Role.EMPLOYEE}

    # Generated colleagues: one manager per other department, then reports.
    managers = {"Engineering": bob, "People": carol}
    generated: list[Employee] = []
    for dept in DEPARTMENTS:
        if dept not in managers:
            managers[dept] = person(
                f"E{len(generated) + 5:04d}",
                fake.first_name(),
                fake.last_name(),
                f"{dept} Manager",
                dept,
                rng.choice(COUNTRIES),
                None,
                rng.randint(3, 8),
            )
            generated.append(managers[dept])
    while len(generated) < 26:
        dept = rng.choice(DEPARTMENTS)
        generated.append(
            person(
                f"E{len(generated) + 5:04d}",
                fake.first_name(),
                fake.last_name(),
                fake.job().split(",")[0][:150],
                dept,
                rng.choice(COUNTRIES),
                managers[dept],
                rng.randint(0, 6),
            )
        )
    session.flush()

    everyone = [*scripted, *generated]
    has_reports = {e.manager_id for e in everyone if e.manager_id}
    for employee in everyone:
        role = scripted.get(employee) or (
            Role.MANAGER if employee.id in has_reports else Role.EMPLOYEE
        )
        session.add(
            User(
                email=employee.email,
                role=role,
                employee=employee,
                password_hash=demo_hash if employee in scripted else unusable_password(),
            )
        )
        low, high = SALARY_RANGE[employee.country]
        session.add(
            Compensation(
                employee_id=employee.id,
                currency=CURRENCY[employee.country],
                base_salary_cents=rng.randrange(low, high, 500)
                * 100,  # $500 steps, stored in cents
                effective_from=date(today.year, 1, 1),
            )
        )
        session.add(
            LeaveBalance(
                employee_id=employee.id,
                leave_type=leave["pto"],
                year=today.year,
                allowed_days=None,
                used_days=Decimal(rng.randint(0, 12)),
            )
        )
        session.add(
            LeaveBalance(
                employee_id=employee.id,
                leave_type=leave["parental"],
                year=today.year,
                allowed_days=Decimal("80"),
                used_days=Decimal("0"),
            )
        )

    def leave_request(
        who: Employee, kind: str, start_in_days: int, days: int, status: RequestStatus, reason: str
    ) -> None:
        start = _next_weekday(today + timedelta(days=start_in_days))
        session.add(
            LeaveRequest(
                employee_id=who.id,
                leave_type=leave[kind],
                start_date=start,
                end_date=_add_working_days(start, days - 1),
                working_days=Decimal(days),
                reason=reason,
                status=status,
                approver_id=who.manager_id,
            )
        )

    leave_request(alice, "pto", -40, 3, RequestStatus.APPROVED, "Family visit")
    leave_request(alice, "pto", 21, 2, RequestStatus.PENDING, "Long weekend")  # waits for Bob
    leave_request(hira, "pto", 10, 1, RequestStatus.PENDING, "Personal errand")
    leave_request(hira, "pto", -15, 2, RequestStatus.REJECTED, "Overlapped with release week")

    def expense(
        who: Employee,
        description: str,
        amount: int,
        currency: str,
        days_ago: int,
        status: RequestStatus,
    ) -> None:
        cents = amount * 100
        usd_cents = int(cents * DEMO_USD_RATE[currency])
        session.add(
            ExpenseClaim(
                employee_id=who.id,
                description=description,
                amount_cents=cents,
                currency=currency,
                amount_usd_cents=usd_cents,
                purchase_date=today - timedelta(days=days_ago),
                is_company_property=usd_cents > 1_000 * 100,  # policy: a physical item over $1,000
                status=status,
                approver_id=who.manager_id,
            )
        )

    expense(alice, "Laptop stand", 89, "USD", 12, RequestStatus.APPROVED)
    expense(bob, "Replacement laptop", 1250, "USD", 5, RequestStatus.PENDING)
    expense(hira, "External monitor", 65_000, "PKR", 20, RequestStatus.PENDING)

    for year in (today.year, today.year + 1):
        for (month, day), name in PK_HOLIDAYS:
            session.add(
                PublicHoliday(
                    country="PK", day=date(year, month, day), name=name, source=HolidaySource.MANUAL
                )
            )

    session.add(
        AuditLog(
            action="db.seed",
            entity="database",
            details={"seed": seed_value, "today": today.isoformat()},
        )
    )
    session.flush()
    return SeedSummary(
        employees=len(everyone),
        users=len(everyone),
        demo_users=[e.email for e in scripted],
        leave_requests=4,
        expense_claims=3,
        holidays=len(PK_HOLIDAYS) * 2,
    )


def _next_weekday(day: date) -> date:
    while day.weekday() >= 5:  # Saturday=5, Sunday=6
        day += timedelta(days=1)
    return day


def _add_working_days(day: date, count: int) -> date:
    """The date `count` working days (Mon-Fri) after `day`. Holidays come later (B4.1)."""
    while count > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count -= 1
    return day
