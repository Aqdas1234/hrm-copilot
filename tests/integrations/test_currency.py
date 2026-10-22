"""Currency conversion with a mocked transport: exact money math, caching, manual rates."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from hrm.db.models import ExchangeRate, RateSource
from hrm.db.seed import seed
from hrm.integrations.currency import CurrencyLookupError, CurrencyService

SATURDAY = date(2026, 10, 3)
FRIDAY = date(2026, 10, 2)


def frankfurter(rate: float, published: date = FRIDAY) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(
        200,
        json={
            "amount": 1.0,
            "base": request.url.params["base"],
            "date": published.isoformat(),
            "rates": {"USD": rate},
        },
    )


def service(
    session: Session, handler: Callable[[httpx.Request], httpx.Response]
) -> tuple[CurrencyService, list[str]]:
    calls: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(counting))
    return CurrencyService(session, client, base_url="https://fx.test/v1", backoff_s=0), calls


def test_usd_needs_no_conversion_or_lookup(session: Session) -> None:
    currency, calls = service(session, frankfurter(1.0))

    result = currency.to_usd(8_900, "usd", on=FRIDAY)

    assert result.usd_cents == 8_900 and result.source is None
    assert calls == []


def test_fetches_the_rate_once_then_uses_the_cache(session: Session) -> None:
    currency, calls = service(session, frankfurter(1.3201))

    first = currency.to_usd(10_000, "GBP", on=FRIDAY)  # £100.00
    second = currency.to_usd(2_500, "GBP", on=FRIDAY)

    assert first.usd_cents == 13_201  # $132.01
    assert second.usd_cents == 3_300  # 2500 x 1.3201 = 3300.25 -> 3300
    assert first.source is RateSource.API
    assert calls == ["https://fx.test/v1/2026-10-02?base=GBP&symbols=USD"]


def test_weekend_uses_the_last_published_rate_and_records_its_date(session: Session) -> None:
    currency, _ = service(session, frankfurter(1.1225, published=FRIDAY))

    result = currency.to_usd(10_000, "EUR", on=SATURDAY)

    assert result.published_on == FRIDAY
    stored = session.scalar(select(ExchangeRate).where(ExchangeRate.currency == "EUR"))
    assert stored is not None and stored.day == SATURDAY and stored.published_on == FRIDAY


@pytest.mark.parametrize(
    ("cents", "rate", "expected"),
    [
        (1, 1.5, 2),
        (3, 0.5, 2),
        (1, 0.4, 0),
        (99_999, 1.08, 107_999),
    ],  # 107998.92 -> 107999; half-up, not banker's
)
def test_rounding_is_half_up_to_whole_cents(
    session: Session, cents: int, rate: float, expected: int
) -> None:
    currency, _ = service(session, frankfurter(rate))

    assert currency.to_usd(cents, "EUR", on=FRIDAY).usd_cents == expected


def test_pkr_uses_the_manual_rate_without_asking_the_api(session: Session) -> None:
    seed(session, today=date(2026, 10, 22), demo_password="x")  # sets PKR manually
    currency, calls = service(session, lambda r: httpx.Response(404, json={"message": "not found"}))

    result = currency.to_usd(6_500_000, "PKR", on=date(2026, 10, 2))  # PKR 65,000

    assert result.usd_cents == 23_400  # $234.00 at 0.0036
    assert result.source is RateSource.MANUAL
    assert calls == []


def test_newest_manual_rate_before_the_date_wins(session: Session) -> None:
    session.add_all(
        [
            ExchangeRate(
                currency="PKR",
                day=date(2026, 1, 1),
                usd_per_unit=Decimal("0.0036"),
                published_on=date(2026, 1, 1),
                source=RateSource.MANUAL,
            ),
            ExchangeRate(
                currency="PKR",
                day=date(2026, 7, 1),
                usd_per_unit=Decimal("0.0035"),
                published_on=date(2026, 7, 1),
                source=RateSource.MANUAL,
            ),
        ]
    )
    currency, _ = service(session, frankfurter(1.0))

    assert currency.to_usd(100_000, "PKR", on=date(2026, 3, 1)).usd_per_unit == Decimal("0.0036")
    assert currency.to_usd(100_000, "PKR", on=date(2026, 9, 1)).usd_per_unit == Decimal("0.0035")


def test_unsupported_currency_without_manual_rate_is_a_clear_error(session: Session) -> None:
    currency, _ = service(session, lambda r: httpx.Response(404, json={"message": "not found"}))

    with pytest.raises(CurrencyLookupError, match="ask Finance"):
        currency.to_usd(10_000, "PKR", on=FRIDAY)


def test_service_down_raises_and_caches_nothing(session: Session) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    currency, calls = service(session, down)

    with pytest.raises(CurrencyLookupError, match="unavailable"):
        currency.to_usd(10_000, "EUR", on=FRIDAY)
    assert len(calls) == 3
    assert session.scalars(select(ExchangeRate)).all() == []


@pytest.mark.network
def test_real_frankfurter_rate_for_euro(session: Session) -> None:
    with httpx.Client(timeout=15) as client:
        result = CurrencyService(session, client).to_usd(10_000, "EUR", on=date(2026, 10, 2))

    assert 9_000 < result.usd_cents < 14_000  # €100 is roughly $100-$130
