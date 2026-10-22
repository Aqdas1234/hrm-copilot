"""Who is asking. Built once per request from the logged-in user (B2.2: from the JWT)."""

from dataclasses import dataclass

from hrm.db.models import Role, User


@dataclass(frozen=True)
class UserContext:
    user_id: int
    employee_id: int
    email: str
    role: Role

    @classmethod
    def from_user(cls, user: User) -> "UserContext":
        return cls(user_id=user.id, employee_id=user.employee_id, email=user.email, role=user.role)

    @property
    def is_hr_admin(self) -> bool:
        return self.role is Role.HR_ADMIN
