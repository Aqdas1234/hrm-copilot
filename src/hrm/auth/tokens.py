"""Login tokens (JWT): signed, short-lived, carrying only who the user is.

The token holds the user id and role plus issue and expiry times, signed with HS256 and a
secret from settings. Anyone can read a JWT's contents, so it never holds anything secret;
the signature makes it impossible to change or forge without the secret.
"""

from datetime import UTC, datetime, timedelta

import jwt

from hrm.core.errors import HRMError
from hrm.db.models import User

ALGORITHM = "HS256"
ISSUER = "hrm-copilot"


class InvalidTokenError(HRMError):
    """Missing, malformed, tampered with, or expired."""


def create_token(user: User, *, secret: str, ttl_minutes: int, now: datetime | None = None) -> str:
    issued = now or datetime.now(UTC)
    claims = {
        "sub": str(user.id),
        "role": user.role.value,
        "iss": ISSUER,
        "iat": issued,
        "exp": issued + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def user_id_from_token(token: str, *, secret: str) -> int:
    """The user id inside a valid token. Raises InvalidTokenError otherwise."""
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],  # never accept "none" or another algorithm
            issuer=ISSUER,
            options={"require": ["sub", "exp", "iat", "iss"]},
        )
        return int(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidTokenError("Invalid or expired token.") from exc
