import pytest

from hrm.auth.passwords import hash_password, unusable_password, verify_password


def test_correct_password_verifies_and_wrong_one_does_not() -> None:
    stored = hash_password("correct horse")

    assert verify_password("correct horse", stored)
    assert not verify_password("correct hors", stored)


def test_same_password_gets_a_different_hash_each_time() -> None:
    assert hash_password("same") != hash_password("same")  # random salt


def test_hash_does_not_contain_the_password() -> None:
    assert "s3cret" not in hash_password("s3cret")


@pytest.mark.parametrize("stored", ["", "plain-text", "bcrypt$x$y$z$a$b", "scrypt$1$2$3$!!$!!"])
def test_malformed_hashes_never_verify(stored: str) -> None:
    assert not verify_password("anything", stored)


def test_unusable_password_rejects_common_guesses() -> None:
    stored = unusable_password()

    assert not any(verify_password(g, stored) for g in ["", "password", "hrm-demo-123"])
