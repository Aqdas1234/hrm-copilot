"""The same schema on a real MySQL server. Skipped by default; run with:

    HRM_TEST_DATABASE_URL=mysql+pymysql://user:pass@127.0.0.1:3306/hrm_test pytest -m mysql

Use a throwaway database: the test drops all tables at the end.
"""

import os
from datetime import date

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import sessionmaker

from hrm.db.models import Base, Department, Employee, EmploymentType, Role, User
from hrm.db.session import create_schema, make_engine

pytestmark = pytest.mark.mysql


@pytest.mark.skipif(
    "HRM_TEST_DATABASE_URL" not in os.environ, reason="HRM_TEST_DATABASE_URL not set"
)
def test_schema_and_a_round_trip_on_mysql() -> None:
    engine = make_engine(os.environ["HRM_TEST_DATABASE_URL"])
    try:
        create_schema(engine)
        assert set(inspect(engine).get_table_names()) >= set(Base.metadata.tables)
        with sessionmaker(engine)() as s:
            employee = Employee(
                employee_code="T1",
                first_name="Test",
                last_name="User",
                email="t1@example.com",
                job_title="Tester",
                department=Department(name="QA"),
                country="PK",
                employment_type=EmploymentType.FULL_TIME,
                start_date=date(2025, 1, 1),
            )
            s.add(
                User(
                    email="t1@example.com", password_hash="h", role=Role.HR_ADMIN, employee=employee
                )
            )
            s.commit()
            assert s.scalar(select(User.role)) is Role.HR_ADMIN
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
