"""Public holidays: the database first, then the Nager.Date API (cached in the database).

- Countries the API covers (US, GB, DE, ...) are fetched once per year and stored.
- Countries it does not cover (e.g. Pakistan, HTTP 204) use holidays entered in the
  database (`source=manual`, see `hrm db seed`).
- Only nationwide holidays are kept: regional ones (e.g. a holiday in some US states)
  would be wrong for most employees, and we do not store employees' regions.
- If the API is down after retries, HolidayLookupError is raised: silently assuming
  "no holidays" would give wrong working-day counts for leave.
"""

from dataclasses import dataclass
from datetime import date

import httpx
from sqlalchemy.orm import Session

from hrm.core.errors import HRMError
from hrm.core.logging import get_logger
from hrm.db import repository as repo
from hrm.db.models import HolidaySource, PublicHoliday
from hrm.integrations.http import ServiceUnavailableError, get_with_retries

log = get_logger(__name__)


class HolidayLookupError(HRMError):
    """Holidays could not be determined (service down, or an unexpected answer)."""


@dataclass(frozen=True)
class Holiday:
    day: date
    name: str
    source: HolidaySource


class HolidayService:
    def __init__(
        self,
        session: Session,
        client: httpx.Client,
        *,
        base_url: str = "https://date.nager.at/api/v3",
        attempts: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._session = session
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._attempts = attempts
        self._backoff_s = backoff_s

    def holidays(self, country: str, year: int) -> list[Holiday]:
        country = country.upper()
        stored = repo.holidays_between(self._session, country, date(year, 1, 1), date(year, 12, 31))
        if stored:
            return [Holiday(h.day, h.name, h.source) for h in stored]

        fetched = self._fetch(country, year)
        repo.add_holidays(
            self._session,
            [
                PublicHoliday(country=country, day=h.day, name=h.name, source=HolidaySource.API)
                for h in fetched
            ],
        )
        log.info("holidays.cached", country=country, year=year, count=len(fetched))
        return fetched

    def is_holiday(self, country: str, day: date) -> bool:
        return any(h.day == day for h in self.holidays(country, day.year))

    def _fetch(self, country: str, year: int) -> list[Holiday]:
        url = f"{self._base_url}/PublicHolidays/{year}/{country}"
        try:
            response = get_with_retries(
                self._client, url, attempts=self._attempts, backoff_s=self._backoff_s
            )
        except ServiceUnavailableError as exc:
            raise HolidayLookupError(f"Public holiday service unavailable for {country}.") from exc
        if response.status_code in (204, 404):  # country not covered / unknown
            log.warning("holidays.country_not_covered", country=country, year=year)
            return []
        if response.status_code != 200:
            raise HolidayLookupError(f"Unexpected answer {response.status_code} for {country}.")
        try:
            return [
                Holiday(date.fromisoformat(item["date"]), item["name"], HolidaySource.API)
                for item in response.json()
                if item.get("global", True)  # nationwide only
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise HolidayLookupError(f"Malformed holiday data for {country}.") from exc
