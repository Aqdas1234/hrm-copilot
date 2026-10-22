"""A stand-in for an external payroll provider (`hrm mock-payroll`).

Like a real provider it receives salary data from HR (here: loaded once from our database)
and serves monthly payslips. It fails with HTTP 503 at a configurable rate, so the
client's retries and circuit breaker are exercised for real. Tax is a flat 22% (demo only).
"""

import random
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TAX_RATE = Decimal("0.22")


class PayslipOut(BaseModel):
    employee_code: str
    period: str  # "2026-09"
    currency: str
    gross_cents: int
    tax_cents: int
    net_cents: int
    paid_on: date


def create_mock_payroll_app(
    yearly_salaries: dict[str, tuple[int, str]],  # employee_code -> (yearly cents, currency)
    *,
    failure_rate: float = 0.2,
    seed: int | None = None,
    today: date | None = None,
) -> FastAPI:
    rng = random.Random(seed)
    app = FastAPI(title="Mock payroll provider")

    @app.get("/payslips/{employee_code}/{year}/{month}", response_model=PayslipOut)
    def payslip(employee_code: str, year: int, month: int) -> PayslipOut:
        if rng.random() < failure_rate:
            raise HTTPException(status_code=503, detail="Payroll system busy, try again.")
        current = today or date.today()
        if employee_code not in yearly_salaries or not 1 <= month <= 12:
            raise HTTPException(status_code=404, detail="No such payslip.")
        if (year, month) >= (current.year, current.month):  # paid at the end of the month
            raise HTTPException(status_code=404, detail="No such payslip.")
        yearly_cents, currency = yearly_salaries[employee_code]
        gross = (Decimal(yearly_cents) / 12).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        tax = (gross * TAX_RATE).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return PayslipOut(
            employee_code=employee_code,
            period=f"{year}-{month:02d}",
            currency=currency,
            gross_cents=int(gross),
            tax_cents=int(tax),
            net_cents=int(gross - tax),
            paid_on=_last_working_day(year, month),
        )

    return app


def _last_working_day(year: int, month: int) -> date:
    day = date(year + month // 12, month % 12 + 1, 1)  # first day of next month
    day = date.fromordinal(day.toordinal() - 1)
    while day.weekday() >= 5:
        day = date.fromordinal(day.toordinal() - 1)
    return day
