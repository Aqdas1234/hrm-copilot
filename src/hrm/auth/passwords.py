"""Password hashing with scrypt (Python standard library).

scrypt is deliberately slow and memory-hard, so stolen hashes are expensive to crack.
Stored format: "scrypt$n$r$p$<salt base64>$<hash base64>", so parameters can be raised
later without breaking existing hashes.
"""

import base64
import hashlib
import hmac
import secrets

_N, _R, _P = 2**14, 8, 1  # cost parameters (memory ~16 MB per hash)
_SALT_BYTES = 16
_KEY_BYTES = 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_KEY_BYTES)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)  # constant time: no timing leak


def unusable_password() -> str:
    """A locked marker no password can match ("!" like a locked Unix account); for accounts
    that must not log in. Cheap to create, unlike hashing a random password."""
    return f"!locked-{secrets.token_urlsafe(16)}"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
