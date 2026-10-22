"""HR tickets: an employee asks HR for help; HR works the ticket until it is closed.

Same rules as leave and expenses:
- anyone opens tickets for themselves and sees only their own;
- only HR admins see the queue and change a ticket's status, never on their own ticket;
- status moves forward only: open -> in_progress -> closed, or open -> closed; closing
  needs a resolution the employee can read; closed is final;
- "not found" and "not allowed" look the same; every action and refusal is audited.
The body is free text from the employee: untrusted wherever it is shown to a model.
"""

from datetime import UTC, datetime
from typing import NoReturn

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError
from hrm.auth.context import UserContext
from hrm.core.errors import HRMError
from hrm.db import repository as repo
from hrm.db.models import HRTicket, TicketCategory, TicketStatus

MAX_SUBJECT = 200
MAX_BODY = 4_000
MAX_RESOLUTION = 2_000
NEXT_STATUS = {
    TicketStatus.OPEN: {TicketStatus.IN_PROGRESS, TicketStatus.CLOSED},
    TicketStatus.IN_PROGRESS: {TicketStatus.CLOSED},
    TicketStatus.CLOSED: set(),
}


class TicketRuleError(HRMError):
    """The ticket change breaks a rule; the message says which."""


class TicketView(BaseModel):
    id: int
    employee_id: int
    category: TicketCategory
    subject: str
    body: str
    status: TicketStatus
    resolution: str | None
    created_at: datetime
    updated_at: datetime | None
    closed_at: datetime | None

    @classmethod
    def of(cls, t: HRTicket) -> "TicketView":
        return cls(
            id=t.id,
            employee_id=t.employee_id,
            category=t.category,
            subject=t.subject,
            body=t.body,
            status=t.status,
            resolution=t.resolution,
            created_at=t.created_at,
            updated_at=t.updated_at,
            closed_at=t.closed_at,
        )


class TicketService:
    def __init__(self, session: Session, ctx: UserContext) -> None:
        self._session = session
        self._ctx = ctx

    def open(self, category: str, subject: str, body: str) -> TicketView:
        ticket = HRTicket(
            employee_id=self._ctx.employee_id,  # always the logged-in user
            category=_category(category),
            subject=_required(subject, "subject", MAX_SUBJECT),
            body=_required(body, "description", MAX_BODY),
            status=TicketStatus.OPEN,
        )
        self._session.add(ticket)
        self._session.flush()
        self._audit("ticket.open", ticket, {"category": ticket.category.value})
        return TicketView.of(ticket)

    def mine(self, status: TicketStatus | None = None) -> list[TicketView]:
        query = select(HRTicket).where(HRTicket.employee_id == self._ctx.employee_id)
        if status is not None:
            query = query.where(HRTicket.status == status)
        return [TicketView.of(t) for t in self._session.scalars(query.order_by(HRTicket.id.desc()))]

    def queue(self) -> list[TicketView]:
        """Tickets HR still has to handle, oldest first (HR admins only)."""
        if not self._ctx.is_hr_admin:
            self._deny("ticket.queue", None)
        query = (
            select(HRTicket)
            .where(HRTicket.status != TicketStatus.CLOSED)
            .order_by(HRTicket.created_at, HRTicket.id)
        )
        return [TicketView.of(t) for t in self._session.scalars(query)]

    def update(
        self, ticket_id: int, status: TicketStatus, resolution: str | None = None
    ) -> TicketView:
        ticket = self._session.get(HRTicket, ticket_id)
        if (
            ticket is None
            or not self._ctx.is_hr_admin
            or ticket.employee_id == self._ctx.employee_id  # nobody handles their own case
        ):
            self._deny("ticket.update", ticket_id)
        if status not in NEXT_STATUS[ticket.status]:
            raise TicketRuleError(f"A {ticket.status.value} ticket cannot move to {status.value}.")
        now = datetime.now(UTC)
        if status is TicketStatus.CLOSED:
            ticket.resolution = _required(resolution, "resolution", MAX_RESOLUTION)
            ticket.closed_at = now
        ticket.status = status
        ticket.updated_at = now
        self._session.flush()
        self._audit("ticket.update", ticket, {"status": status.value})
        return TicketView.of(ticket)

    def _audit(self, action: str, ticket: HRTicket, details: dict[str, object]) -> None:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action=action,
            entity="hr_ticket",
            entity_id=ticket.id,
            details=details,
        )

    def _deny(self, action: str, ticket_id: int | None) -> NoReturn:
        repo.record_audit(
            self._session,
            actor_user_id=self._ctx.user_id,
            action="access.denied",
            entity="hr_ticket",
            entity_id=ticket_id,
            details={"attempted": action},
        )
        raise AccessDeniedError("Ticket not found or not accessible.")


def _category(value: str) -> TicketCategory:
    try:
        return TicketCategory(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(c.value for c in TicketCategory)
        raise TicketRuleError(f"Unknown category {value!r}; use one of: {choices}.") from exc


def _required(value: str | None, what: str, limit: int) -> str:
    text = (value or "").strip()
    if not text:
        raise TicketRuleError(f"Please give a {what}.")
    if len(text) > limit:
        raise TicketRuleError(f"The {what} is too long (at most {limit} characters).")
    return text
