"""HR endpoints that need a logged-in user: login, own data, team, employees.

Every request: read the bearer token, reload the user from the database (so a disabled
account or a changed role takes effect immediately), then go through HRAccess, the
permission layer. Refusals and failed logins are audited, and those audit entries are
committed even though the request fails.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from hrm.auth.access import AccessDeniedError, EmployeeView, HRAccess, LeaveRequestView, SalaryView
from hrm.auth.context import UserContext
from hrm.auth.login import InvalidCredentialsError, authenticate
from hrm.auth.tokens import InvalidTokenError, create_token, user_id_from_token
from hrm.core.errors import ConfigurationError
from hrm.db import repository as repo
from hrm.db.models import User

router = APIRouter()
MIN_SECRET_LENGTH = 32
_bearer = HTTPBearer(auto_error=False)

# Failures whose audit entries must be kept although the request fails.
_AUDITED_FAILURES = (AccessDeniedError, InvalidCredentialsError)


@dataclass(frozen=True)
class AuthConfig:
    secret: str | None
    ttl_minutes: int

    def require_secret(self) -> str:
        if not self.secret:
            raise ConfigurationError("HRM_JWT_SECRET is not set; login is disabled.")
        if len(self.secret) < MIN_SECRET_LENGTH:  # RFC 7518: HS256 keys of at least 32 bytes
            raise ConfigurationError("HRM_JWT_SECRET is too short; use at least 32 characters.")
        return self.secret


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


def get_session(request: Request) -> Iterator[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    session = factory()
    try:
        yield session
        session.commit()
    except _AUDITED_FAILURES:
        session.commit()  # keep the audit entry that records the refusal
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def current_user(
    request: Request,
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> UserContext:
    if credentials is None:
        raise InvalidTokenError("Missing bearer token.")
    config: AuthConfig = request.app.state.auth
    user_id = user_id_from_token(credentials.credentials, secret=config.require_secret())
    user = session.get(User, user_id)
    if user is None or not user.is_active:
        raise InvalidTokenError("Account not found or disabled.")
    return UserContext.from_user(user)


def hr_access(session: SessionDep, ctx: Annotated[UserContext, Depends(current_user)]) -> HRAccess:
    return HRAccess(session, ctx)


AccessDep = Annotated[HRAccess, Depends(hr_access)]


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest, request: Request, session: SessionDep) -> TokenResponse:
    config: AuthConfig = request.app.state.auth
    secret = config.require_secret()
    try:
        user = authenticate(session, body.email, body.password)
    except InvalidCredentialsError:
        repo.record_audit(
            session,
            actor_user_id=None,
            action="auth.login_failed",
            entity="user",
            details={"email": body.email.strip().lower()},
        )
        raise
    repo.record_audit(
        session, actor_user_id=user.id, action="auth.login", entity="user", entity_id=user.id
    )
    token = create_token(user, secret=secret, ttl_minutes=config.ttl_minutes)
    return TokenResponse(access_token=token, expires_in=config.ttl_minutes * 60)


@router.get("/me", response_model=EmployeeView, tags=["me"])
def me(access: AccessDep) -> EmployeeView:
    return access.me()


@router.get("/me/leave", response_model=list[LeaveRequestView], tags=["me"])
def my_leave(access: AccessDep) -> list[LeaveRequestView]:
    return access.leave_requests(access.me().id)


@router.get("/me/salary", response_model=SalaryView, tags=["me"])
def my_salary(access: AccessDep) -> SalaryView:
    return access.salary(access.me().id, reason="self-service")


@router.get("/team", response_model=list[EmployeeView], tags=["team"])
def my_team(access: AccessDep) -> list[EmployeeView]:
    return access.my_team()


@router.get("/team/approvals", response_model=list[LeaveRequestView], tags=["team"])
def pending_approvals(access: AccessDep) -> list[LeaveRequestView]:
    return access.pending_approvals()


@router.get("/employees/{employee_id}", response_model=EmployeeView, tags=["employees"])
def employee(employee_id: int, access: AccessDep) -> EmployeeView:
    return access.employee(employee_id)


@router.get("/employees/{employee_id}/salary", response_model=SalaryView, tags=["employees"])
def employee_salary(
    employee_id: int,
    access: AccessDep,
    reason: Annotated[
        str, Query(min_length=3, max_length=200, description="Recorded in the audit log.")
    ],
) -> SalaryView:
    return access.salary(employee_id, reason=reason)
