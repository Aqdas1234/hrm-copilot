"""Currency conversion to USD for expense claims.

Lookup order for a currency on a given day:
  1. USD: no conversion.
  2. A rate already fetched for that day (cached in the database).
  3. The newest manual rate set on or before that day: Finance sets these for currencies
     the API does not cover (e.g. PKR), and they take precedence over the API.
  4. The Frankfurter API (European Central Bank rates), then cached. On weekends and
     holidays it returns the last published rate; that publication date is stored too.
No rate found means an error, never a guessed rate.
Money is calculated with Decimal and rounded half-up to whole cents.
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import httpx
from sqlalchemy.orm import Session

from hrm.core.errors import HRMError
from hrm.core.logging import get_logger
from hrm.db import repository as repo
from hrm.db.models import ExchangeRate, RateSource
from hrm.integrations.http import ServiceUnavailableError, get_with_retries

log = get_logger(__name__)


class CurrencyLookupError(HRMError):
    """No rate available for the currency and date (unsupported, or service down)."""


@dataclass(frozen=True)
class Conversion:
    amount_cents: int
    currency: str
    usd_cents: int
    usd_per_unit: Decimal
    published_on: date  # the day the rate was published (may be before the requested day)
    source: RateSource | None  # None for USD (no conversion)


class CurrencyService:
    def __init__(
        self,
        session: Session,
        client: httpx.Client,
        *,
        base_url: str = "https://api.frankfurter.dev/v1",
        attempts: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._session = session
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._attempts = attempts
        self._backoff_s = backoff_s

    def to_usd(self, amount_cents: int, currency: str, on: date) -> Conversion:
        currency = currency.upper()
        if currency == "USD":
            return Conversion(amount_cents, "USD", amount_cents, Decimal(1), on, None)
        rate = self._rate(currency, on)
        usd_cents = (Decimal(amount_cents) * rate.usd_per_unit).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
        return Conversion(
            amount_cents,
            currency,
            int(usd_cents),
            rate.usd_per_unit,
            rate.published_on,
            rate.source,
        )

    def _rate(self, currency: str, on: date) -> ExchangeRate:
        rate = repo.cached_rate(self._session, currency, on) or repo.manual_rate(
            self._session, currency, on
        )
        if rate is not None:
            return rate
        rate = self._fetch(currency, on)
        repo.add_rate(self._session, rate)
        log.info(
            "currency.cached", currency=currency, day=on.isoformat(), rate=str(rate.usd_per_unit)
        )
        return rate

    def _fetch(self, currency: str, on: date) -> ExchangeRate:
        url = f"{self._base_url}/{on.isoformat()}?base={currency}&symbols=USD"
        try:
            response = get_with_retries(
                self._client, url, attempts=self._attempts, backoff_s=self._backoff_s
            )
        except ServiceUnavailableError as exc:
            raise CurrencyLookupError("The currency rate service is unavailable.") from exc
        if response.status_code == 404:
            raise CurrencyLookupError(
                f"No exchange rate for {currency}; ask Finance to set a manual rate."
            )
        if response.status_code != 200:
            raise CurrencyLookupError(f"Unexpected answer {response.status_code} for {currency}.")
        try:
            body = response.json()
            return ExchangeRate(
                currency=currency,
                day=on,
                usd_per_unit=Decimal(str(body["rates"]["USD"])),
                published_on=date.fromisoformat(body["date"]),
                source=RateSource.API,
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise CurrencyLookupError(f"Malformed rate data for {currency}.") from exc
