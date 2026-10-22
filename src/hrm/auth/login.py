"""Checking an email and password.

Unknown email and wrong password give the same error, and take about the same time
(a dummy hash is checked when the email is unknown), so attackers cannot find out which
emails exist.
"""

from sqlalchemy.orm import Session

from hrm.auth.passwords import hash_password, verify_password
from hrm.core.errors import HRMError
from hrm.db import repository as repo
from hrm.db.models import User

_DUMMY_HASH = hash_password("timing-equalizer")


class InvalidCredentialsError(HRMError):
    """Wrong email or password, or the account is disabled."""


def authenticate(session: Session, email: str, password: str) -> User:
    user = repo.get_user_by_email(session, email)
    if user is None:
        verify_password(password, _DUMMY_HASH)  # same work as a real check
        raise InvalidCredentialsError("Invalid email or password.")
    if not verify_password(password, user.password_hash) or not user.is_active:
        raise InvalidCredentialsError("Invalid email or password.")
    return user
