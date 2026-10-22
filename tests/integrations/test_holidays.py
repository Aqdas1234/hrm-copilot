"""Holiday lookups with a mocked HTTP transport: no network, every request counted."""

from collections.abc import Callable
from datetime import date

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.db.models import HolidaySource, PublicHoliday
from hrm.db.seed import seed
from hrm.integrations.holidays import HolidayLookupError, HolidayService
from hrm.integrations.http import ServiceUnavailableError, get_with_retries

US_2026 = [
    {"date": "2026-01-01", "name": "New Year's Day", "global": True},
    {"date": "2026-02-12", "name": "Lincoln's Birthday", "global": False, "counties": ["US-CA"]},
    {"date": "2026-07-03", "name": "Independence Day", "global": True},
]


def service(
    session: Session, handler: Callable[[httpx.Request], httpx.Response]
) -> tuple[HolidayService, list[str]]:
    calls: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(counting))
    return HolidayService(session, client, base_url="https://api.test/v3", backoff_s=0), calls


def test_fetches_nationwide_holidays_once_then_uses_the_database(session: Session) -> None:
    holidays, calls = service(session, lambda r: httpx.Response(200, json=US_2026))

    first = holidays.holidays("us", 2026)
    second = holidays.holidays("US", 2026)

    assert [h.name for h in first] == ["New Year's Day", "Independence Day"]  # regional one skipped
    assert all(h.source is HolidaySource.API for h in first)
    assert [h.day for h in second] == [h.day for h in first]
    assert calls == ["/v3/PublicHolidays/2026/US"]  # second call answered from the database


def test_country_not_covered_by_the_api_uses_manual_database_holidays(session: Session) -> None:
    seed(session, today=date(2026, 10, 22), demo_password="x")  # stores Pakistan's holidays
    holidays, calls = service(session, lambda r: httpx.Response(204))

    pk = holidays.holidays("PK", 2026)

    assert "Independence Day" in [h.name for h in pk]
    assert all(h.source is HolidaySource.MANUAL for h in pk)
    assert calls == []  # never asked the API


@pytest.mark.parametrize("status", [204, 404])
def test_uncovered_country_without_manual_holidays_returns_none(
    session: Session, status: int
) -> None:
    holidays, _ = service(session, lambda r: httpx.Response(status))

    assert holidays.holidays("XX", 2026) == []


def test_temporary_failures_are_retried(session: Session) -> None:
    answers = iter([httpx.Response(503), httpx.Response(200, json=US_2026)])
    holidays, calls = service(session, lambda r: next(answers))

    assert len(holidays.holidays("US", 2026)) == 2
    assert len(calls) == 2


def test_service_down_raises_instead_of_assuming_no_holidays(session: Session) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    holidays, calls = service(session, down)

    with pytest.raises(HolidayLookupError):
        holidays.holidays("US", 2026)
    assert len(calls) == 3  # all attempts used
    assert session.scalars(select(PublicHoliday)).all() == []  # nothing cached


def test_malformed_answer_is_an_error(session: Session) -> None:
    holidays, _ = service(session, lambda r: httpx.Response(200, json=[{"oops": True}]))

    with pytest.raises(HolidayLookupError, match="Malformed"):
        holidays.holidays("US", 2026)


def test_is_holiday(session: Session) -> None:
    holidays, _ = service(session, lambda r: httpx.Response(200, json=US_2026))

    assert holidays.is_holiday("US", date(2026, 7, 3))
    assert not holidays.is_holiday("US", date(2026, 7, 6))


# --- the retry helper -----------------------------------------------------------------------


def test_client_errors_are_not_retried() -> None:
    calls: list[int] = []

    def not_found(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(not_found))

    assert get_with_retries(client, "https://api.test/x", sleep=lambda s: None).status_code == 404
    assert len(calls) == 1


def test_backoff_doubles_between_attempts() -> None:
    waits: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))

    with pytest.raises(ServiceUnavailableError):
        get_with_retries(
            client, "https://api.test/x", attempts=3, backoff_s=0.5, sleep=waits.append
        )

    assert waits == [0.5, 1.0]  # no wait after the last attempt


# --- the real API (pytest -m network) ------------------------------------------------------


@pytest.mark.network
def test_real_nager_api_for_germany(session: Session) -> None:
    with httpx.Client(timeout=15) as client:
        found = HolidayService(session, client).holidays("DE", 2026)

    assert date(2026, 12, 25) in [h.day for h in found]
