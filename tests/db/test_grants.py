import pytest

from hrm.analytics.schema import VIEW_NAMES
from hrm.core.errors import ConfigurationError
from hrm.db.grants import analytics_grants, grant_analytics_access
from hrm.db.session import make_engine


def test_one_select_grant_per_analytics_view_and_nothing_else() -> None:
    statements = analytics_grants("hrm")
    assert len(statements) == len(VIEW_NAMES)
    assert all(s.startswith("GRANT SELECT ON `hrm`.`v_") for s in statements)
    assert all(s.endswith("TO 'hrm_analytics'@'%'") for s in statements)


def test_grants_are_mysql_only() -> None:
    with pytest.raises(ConfigurationError):
        grant_analytics_access(make_engine("sqlite://"), "hrm")
