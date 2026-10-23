"""HR data model.

Design rules:
- Salary lives in its own `compensation` table, so ordinary employee queries cannot leak it.
- Money is stored as integer cents with an ISO currency code (no floating-point rounding).
- Enums are stored as short strings with a CHECK constraint, which works on MySQL and SQLite.
- The audit log is append-only: the code never updates or deletes its rows.
- Timestamps are UTC.
"""

import enum
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Predictable constraint names, so migrations and error messages are readable.
NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def utcnow() -> datetime:
    return datetime.now(UTC)


def _enum(kind: type[enum.Enum], name: str) -> Enum:
    """A portable enum column: VARCHAR + CHECK, storing the enum's value."""
    return Enum(
        kind,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [m.value for m in members],
        length=20,
    )


class Role(enum.StrEnum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR_ADMIN = "hr_admin"


class EmploymentType(enum.StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACTOR = "contractor"


class RequestStatus(enum.StrEnum):
    PENDING = "pending"
    # Expense claims only: the manager approved, HR still has to. Leave never uses it, and
    # the leave table's CHECK constraint does not allow it (migration 0004).
    PENDING_HR = "pending_hr"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class TicketStatus(enum.StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class TicketCategory(enum.StrEnum):
    PAYROLL = "payroll"
    BENEFITS = "benefits"
    LEAVE = "leave"
    ACCESS = "access"  # systems, badges, accounts
    OTHER = "other"


class HolidaySource(enum.StrEnum):
    API = "api"  # fetched from Nager.Date and cached
    MANUAL = "manual"  # entered for countries the API does not cover


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class Employee(Base):
    """Who someone is in the company. Deliberately has no salary field."""

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_code: Mapped[str] = mapped_column(String(20), unique=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    job_title: Mapped[str] = mapped_column(String(150))
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"))
    manager_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"))
    country: Mapped[str] = mapped_column(String(2))  # ISO 3166-1 alpha-2, e.g. "US", "PK"
    employment_type: Mapped[EmploymentType] = mapped_column(
        _enum(EmploymentType, "employment_type")
    )
    start_date: Mapped[date] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(default=True)

    department: Mapped[Department] = relationship()
    manager: Mapped["Employee | None"] = relationship(remote_side=[id], back_populates="reports")
    reports: Mapped[list["Employee"]] = relationship(back_populates="manager")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class User(Base):
    """A login. Role decides what the person may see and do."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(_enum(Role, "role"))
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    employee: Mapped[Employee] = relationship()


class Compensation(Base):
    """Salary history, kept apart from Employee. Only HR admins may read it."""

    __tablename__ = "compensation"
    __table_args__ = (CheckConstraint("base_salary_cents >= 0", name="salary_not_negative"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    base_salary_cents: Mapped[int] = mapped_column(BigInteger)  # yearly, in cents
    currency: Mapped[str] = mapped_column(String(3))  # ISO 4217, e.g. "USD"
    effective_from: Mapped[date] = mapped_column(Date)


class LeaveType(Base):
    __tablename__ = "leave_types"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True)  # "pto", "sick", "parental"
    name: Mapped[str] = mapped_column(String(100))
    # Working days per year; None means no set limit (Sourcegraph's flexible PTO).
    annual_days: Mapped[Decimal | None] = mapped_column(Numeric(5, 1))
    paid: Mapped[bool] = mapped_column(default=True)


class LeaveBalance(Base):
    """Days used (and allowed, if limited) per employee, leave type and year."""

    __tablename__ = "leave_balances"
    __table_args__ = (
        UniqueConstraint("employee_id", "leave_type_id", "year", name="uq_leave_balance"),
        CheckConstraint("used_days >= 0", name="used_not_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"))
    leave_type_id: Mapped[int] = mapped_column(ForeignKey("leave_types.id"))
    year: Mapped[int]
    allowed_days: Mapped[Decimal | None] = mapped_column(Numeric(5, 1))
    used_days: Mapped[Decimal] = mapped_column(Numeric(5, 1), default=Decimal("0"))

    leave_type: Mapped[LeaveType] = relationship()


class LeaveRequest(Base):
    __tablename__ = "leave_requests"
    __table_args__ = (CheckConstraint("end_date >= start_date", name="dates_in_order"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    leave_type_id: Mapped[int] = mapped_column(ForeignKey("leave_types.id"))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    working_days: Mapped[Decimal] = mapped_column(Numeric(5, 1))
    reason: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[RequestStatus] = mapped_column(
        _enum(RequestStatus, "leave_status"), default=RequestStatus.PENDING
    )
    approver_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    leave_type: Mapped[LeaveType] = relationship()


class PublicHoliday(Base):
    __tablename__ = "public_holidays"
    __table_args__ = (UniqueConstraint("country", "day", name="uq_holiday_country_day"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    country: Mapped[str] = mapped_column(String(2))
    day: Mapped[date] = mapped_column(Date)
    name: Mapped[str] = mapped_column(String(150))
    source: Mapped[HolidaySource] = mapped_column(_enum(HolidaySource, "holiday_source"))


class ExpenseClaim(Base):
    __tablename__ = "expense_claims"
    __table_args__ = (CheckConstraint("amount_cents > 0", name="amount_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    description: Mapped[str] = mapped_column(String(500))
    amount_cents: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3))
    amount_usd_cents: Mapped[int] = mapped_column(BigInteger)  # converted at submission time
    purchase_date: Mapped[date] = mapped_column(Date)
    # Policy: a single physical item over $1,000 becomes company property.
    is_company_property: Mapped[bool] = mapped_column(default=False)
    status: Mapped[RequestStatus] = mapped_column(
        _enum(RequestStatus, "expense_status"), default=RequestStatus.PENDING
    )
    approver_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"))
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class HRTicket(Base):
    __tablename__ = "hr_tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    category: Mapped[TicketCategory] = mapped_column(_enum(TicketCategory, "ticket_category"))
    subject: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)  # free text from the employee: untrusted
    status: Mapped[TicketStatus] = mapped_column(
        _enum(TicketStatus, "ticket_status"), default=TicketStatus.OPEN
    )
    resolution: Mapped[str | None] = mapped_column(Text)  # required to close
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RateSource(enum.StrEnum):
    API = "api"  # fetched from Frankfurter (ECB rates) and cached
    MANUAL = "manual"  # set by Finance for currencies the API does not cover


class ExchangeRate(Base):
    """USD value of one unit of a currency. API rows: one per requested day (with the date
    the rate was actually published). Manual rows: valid from `day` until a newer one."""

    __tablename__ = "exchange_rates"
    __table_args__ = (
        UniqueConstraint("currency", "day", "source", name="uq_rate_currency_day_source"),
        CheckConstraint("usd_per_unit > 0", name="rate_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    currency: Mapped[str] = mapped_column(String(3))
    day: Mapped[date] = mapped_column(Date)
    usd_per_unit: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    published_on: Mapped[date] = mapped_column(Date)
    source: Mapped[RateSource] = mapped_column(_enum(RateSource, "rate_source"))


class AuditLog(Base):
    """Who did or saw what, when. Append-only: rows are only ever inserted."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), primary_key=True
    )
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(50))  # e.g. "leave.request", "salary.read"
    entity: Mapped[str] = mapped_column(String(50))  # e.g. "leave_request", "employee"
    entity_id: Mapped[int | None]
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
