"""Circuit breaker (fake clock) and the payroll client against the real mock payroll app."""

from contextlib import nullcontext
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

from hrm.integrations.circuit_breaker import CircuitBreaker, CircuitOpenError, State
from hrm.integrations.mock_payroll import create_mock_payroll_app
from hrm.integrations.payroll import PayrollClient, PayrollUnavailableError, PayslipNotFoundError

TODAY = date(2026, 10, 22)
SALARIES = {"E0003": (15_850_000, "USD"), "E0004": (540_000_000, "PKR")}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def state(breaker: CircuitBreaker) -> State:
    """Read the state through a call: it changes over time, which mypy cannot see."""
    return breaker.state


def boom() -> None:
    raise RuntimeError("service failed")


# --- the circuit breaker ------------------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_then_rejects_without_calling() -> None:
    breaker = CircuitBreaker("svc", failure_threshold=3, reset_after_s=30, clock=FakeClock())
    calls: list[int] = []

    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(boom)

    assert state(breaker) is State.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: calls.append(1))
    assert calls == []  # the service was not called


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker("svc", failure_threshold=3, clock=FakeClock())

    for outcome in [boom, boom, lambda: "ok", boom, boom]:
        with pytest.raises(RuntimeError) if outcome is boom else nullcontext():
            breaker.call(outcome)

    assert breaker.state is State.CLOSED  # never 3 failures in a row


def test_half_open_trial_success_closes_and_failure_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("svc", failure_threshold=1, reset_after_s=30, clock=clock)
    with pytest.raises(RuntimeError):
        breaker.call(boom)

    clock.now += 30
    assert state(breaker) is State.HALF_OPEN
    with pytest.raises(RuntimeError):
        breaker.call(boom)  # trial failed
    assert state(breaker) is State.OPEN

    clock.now += 30
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.state is State.CLOSED


def test_errors_that_are_not_failures_do_not_count() -> None:
    breaker = CircuitBreaker("svc", failure_threshold=1, clock=FakeClock())

    empty: dict[str, int] = {}

    with pytest.raises(KeyError):
        breaker.call(lambda: empty["missing"], is_failure=lambda e: not isinstance(e, KeyError))

    assert breaker.state is State.CLOSED


# --- the payroll client against the mock provider ---------------------------------------


def payroll(
    failure_rate: float, clock: FakeClock | None = None
) -> tuple[PayrollClient, CircuitBreaker]:
    app = create_mock_payroll_app(SALARIES, failure_rate=failure_rate, seed=7, today=TODAY)
    breaker = CircuitBreaker(
        "payroll", failure_threshold=2, reset_after_s=60, clock=clock or FakeClock()
    )
    client = PayrollClient(
        TestClient(app), base_url="http://payroll.test", breaker=breaker, backoff_s=0
    )
    return client, breaker


def test_healthy_service_returns_the_payslip_with_exact_money() -> None:
    client, _ = payroll(failure_rate=0)

    slip = client.payslip("E0003", 2026, 9)

    assert slip.gross_cents == 1_320_833  # 158,500.00 / 12 = 13,208.33
    assert slip.tax_cents == 290_583  # 22%, rounded half-up
    assert slip.net_cents == slip.gross_cents - slip.tax_cents
    assert slip.paid_on == date(2026, 9, 30) and slip.currency == "USD"


def test_flaky_service_succeeds_thanks_to_retries() -> None:
    slip = {
        "employee_code": "E0004",
        "period": "2026-09",
        "currency": "PKR",
        "gross_cents": 45_000_000,
        "tax_cents": 9_900_000,
        "net_cents": 35_100_000,
        "paid_on": "2026-09-30",
    }
    answers = iter([httpx.Response(503), httpx.Response(503), httpx.Response(200, json=slip)])
    breaker = CircuitBreaker("payroll", failure_threshold=2, clock=FakeClock())
    flaky = httpx.Client(transport=httpx.MockTransport(lambda r: next(answers)))
    client = PayrollClient(flaky, base_url="http://payroll.test", breaker=breaker, backoff_s=0)

    assert client.payslip("E0004", 2026, 9).net_cents == 35_100_000  # third attempt succeeded
    assert breaker.state is State.CLOSED


def test_service_down_opens_the_circuit_then_fails_fast() -> None:
    client, breaker = payroll(failure_rate=1.0)

    for _ in range(2):  # each call already retried 3 times
        with pytest.raises(PayrollUnavailableError):
            client.payslip("E0003", 2026, 9)

    assert breaker.state is State.OPEN
    with pytest.raises(PayrollUnavailableError, match="temporarily unavailable"):
        client.payslip("E0003", 2026, 9)  # rejected without calling the service


def test_circuit_closes_again_when_the_service_recovers() -> None:
    clock = FakeClock()
    down, breaker = payroll(failure_rate=1.0, clock=clock)
    for _ in range(2):
        with pytest.raises(PayrollUnavailableError):
            down.payslip("E0003", 2026, 9)

    healthy_app = create_mock_payroll_app(SALARIES, failure_rate=0, today=TODAY)
    healthy = PayrollClient(
        TestClient(healthy_app), base_url="http://payroll.test", breaker=breaker
    )
    clock.now += 60

    assert healthy.payslip("E0003", 2026, 9).period == "2026-09"
    assert breaker.state is State.CLOSED


@pytest.mark.parametrize(
    ("code", "year", "month"),
    [("E9999", 2026, 9), ("E0003", 2026, 10), ("E0003", 2027, 1)],
    ids=["unknown employee", "current month not paid yet", "future month"],
)
def test_missing_payslips_are_not_counted_as_service_failures(
    code: str, year: int, month: int
) -> None:
    client, breaker = payroll(failure_rate=0)

    for _ in range(3):
        with pytest.raises(PayslipNotFoundError):
            client.payslip(code, year, month)

    assert breaker.state is State.CLOSED
