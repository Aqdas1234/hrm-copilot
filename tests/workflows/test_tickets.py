"""HR tickets: who may do what, the status flow, and the audit trail."""

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.auth.access import AccessDeniedError
from hrm.auth.context import UserContext
from hrm.db import repository as repo
from hrm.db.models import AuditLog, TicketStatus
from hrm.db.seed import seed
from hrm.workflows.tickets import TicketRuleError, TicketService


@pytest.fixture
def db(session: Session) -> Session:
    seed(session, today=date(2026, 10, 22), demo_password="x")
    return session


def as_user(db: Session, name: str) -> TicketService:
    user = repo.get_user_by_email(db, f"{name}@example.com")
    assert user is not None
    return TicketService(db, UserContext.from_user(user))


def open_ticket(db: Session, name: str = "alice.park") -> int:
    return as_user(db, name).open("payroll", "Wrong tax code", "My October payslip ...").id


def test_anyone_opens_a_ticket_for_themselves(db: Session) -> None:
    ticket = as_user(db, "alice.park").open(" Payroll ", "Wrong tax code", "Details")

    assert ticket.category == "payroll"
    assert ticket.status is TicketStatus.OPEN
    assert [t.id for t in as_user(db, "alice.park").mine()] == [ticket.id]
    assert as_user(db, "hira.khan").mine() == []  # only your own


@pytest.mark.parametrize(
    ("category", "subject", "body", "message"),
    [
        ("lunch", "s", "b", "Unknown category"),
        ("payroll", "  ", "b", "give a subject"),
        ("payroll", "s", "", "give a description"),
        ("payroll", "s" * 201, "b", "too long"),
    ],
)
def test_bad_tickets_are_refused(
    db: Session, category: str, subject: str, body: str, message: str
) -> None:
    with pytest.raises(TicketRuleError, match=message):
        as_user(db, "alice.park").open(category, subject, body)


def test_only_hr_admins_see_the_queue(db: Session) -> None:
    ticket_id = open_ticket(db)
    assert [t.id for t in as_user(db, "carol.reyes").queue()] == [ticket_id]
    for name in ("alice.park", "bob.chen"):
        with pytest.raises(AccessDeniedError):
            as_user(db, name).queue()


def test_hr_works_a_ticket_to_closed_with_a_resolution(db: Session) -> None:
    ticket_id = open_ticket(db)
    carol = as_user(db, "carol.reyes")

    carol.update(ticket_id, TicketStatus.IN_PROGRESS)
    with pytest.raises(TicketRuleError, match="give a resolution"):
        carol.update(ticket_id, TicketStatus.CLOSED)
    closed = carol.update(ticket_id, TicketStatus.CLOSED, "Tax code fixed from November.")

    assert closed.status is TicketStatus.CLOSED
    assert closed.resolution == "Tax code fixed from November."
    assert closed.closed_at is not None
    assert as_user(db, "carol.reyes").queue() == []


def test_closed_is_final_and_status_only_moves_forward(db: Session) -> None:
    ticket_id = open_ticket(db)
    carol = as_user(db, "carol.reyes")
    carol.update(ticket_id, TicketStatus.CLOSED, "Done.")
    with pytest.raises(TicketRuleError, match="closed ticket cannot move to in_progress"):
        carol.update(ticket_id, TicketStatus.IN_PROGRESS)
    with pytest.raises(TicketRuleError, match="cannot move to open"):
        carol.update(open_ticket(db), TicketStatus.OPEN)


def test_nobody_else_may_update_and_unknown_looks_the_same(db: Session) -> None:
    ticket_id = open_ticket(db)
    for name, target in (("alice.park", ticket_id), ("bob.chen", ticket_id), ("carol.reyes", 999)):
        with pytest.raises(AccessDeniedError, match="not found or not accessible"):
            as_user(db, name).update(target, TicketStatus.IN_PROGRESS)


def test_hr_admins_cannot_handle_their_own_ticket(db: Session) -> None:
    own = open_ticket(db, "carol.reyes")
    with pytest.raises(AccessDeniedError):
        as_user(db, "carol.reyes").update(own, TicketStatus.CLOSED, "Self-approved.")


def test_actions_and_refusals_are_audited(db: Session) -> None:
    ticket_id = open_ticket(db)
    as_user(db, "carol.reyes").update(ticket_id, TicketStatus.IN_PROGRESS)
    with pytest.raises(AccessDeniedError):
        as_user(db, "bob.chen").update(ticket_id, TicketStatus.CLOSED, "x")

    actions = db.scalars(
        select(AuditLog.action).where(AuditLog.entity == "hr_ticket").order_by(AuditLog.id)
    ).all()
    assert actions == ["ticket.open", "ticket.update", "access.denied"]
