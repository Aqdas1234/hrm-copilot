"""Client for the payroll provider: retries per call, wrapped in a circuit breaker.

"Payslip not found" is a normal answer, not a service fault, so it does not count towards
opening the circuit. Everything else (timeouts, repeated 5xx, bad data) does.
"""

from datetime import date

from pydantic import BaseModel, ValidationError

from hrm.core.errors import HRMError
from hrm.integrations.circuit_breaker import CircuitBreaker, CircuitOpenError
from hrm.integrations.http import HTTPGetter, ServiceUnavailableError, get_with_retries


class PayrollUnavailableError(HRMError):
    """Payroll cannot be reached right now (failing, or the circuit is open)."""


class PayslipNotFoundError(HRMError):
    """No payslip for that employee and month (e.g. the month is not paid yet)."""


class Payslip(BaseModel):
    employee_code: str
    period: str
    currency: str
    gross_cents: int
    tax_cents: int
    net_cents: int
    paid_on: date


class PayrollClient:
    def __init__(
        self,
        client: HTTPGetter,
        *,
        base_url: str,
        breaker: CircuitBreaker | None = None,
        attempts: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._breaker = breaker or CircuitBreaker("payroll")
        self._attempts = attempts
        self._backoff_s = backoff_s

    def payslip(self, employee_code: str, year: int, month: int) -> Payslip:
        try:
            return self._breaker.call(
                lambda: self._fetch(employee_code, year, month),
                is_failure=lambda exc: not isinstance(exc, PayslipNotFoundError),
            )
        except CircuitOpenError as exc:
            raise PayrollUnavailableError("Payroll is temporarily unavailable.") from exc

    def _fetch(self, employee_code: str, year: int, month: int) -> Payslip:
        url = f"{self._base_url}/payslips/{employee_code}/{year}/{month}"
        try:
            response = get_with_retries(
                self._client, url, attempts=self._attempts, backoff_s=self._backoff_s
            )
        except ServiceUnavailableError as exc:
            raise PayrollUnavailableError("Payroll is temporarily unavailable.") from exc
        if response.status_code == 404:
            raise PayslipNotFoundError(f"No payslip for {year}-{month:02d}.")
        if response.status_code != 200:
            raise PayrollUnavailableError(f"Unexpected payroll answer {response.status_code}.")
        try:
            return Payslip.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise PayrollUnavailableError("Payroll returned malformed data.") from exc
