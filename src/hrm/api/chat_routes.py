"""Chat with the HR agent: send a message, confirm a pending action, read a conversation.

    POST /chat/{conversation_id}/messages   {"message": "..."}  -> answer or "confirm"
    POST /chat/{conversation_id}/confirm    {"approved": true}  -> answer
    GET  /chat/{conversation_id}                                -> messages + pending action

The client picks the conversation id; the server combines it with the logged-in user
(agents/memory.py), so nobody can open someone else's conversation.
Each request is one database transaction, and each confirmed action commits on its own
(agents/tools.py), so the database and the conversation always agree.
"""

from datetime import date
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Path, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from hrm.agents.memory import CONVERSATION_ID_PATTERN, thread_id
from hrm.agents.runner import AgentReply, AgentRunner, ChatMessage
from hrm.api.hr_routes import SessionDep, current_user
from hrm.api.rate_limit import RateLimiter
from hrm.auth.context import UserContext
from hrm.core.errors import ConfigurationError

router = APIRouter(prefix="/chat", tags=["chat"])

# A hard cap on the request body; the input guardrail gives the friendly limit (2,000).
MAX_MESSAGE_CHARS = 8_000


class RunnerFactory(Protocol):
    """AgentFactory, or a test double."""

    def runner(
        self, session: Session, user: UserContext, today: date | None = None
    ) -> AgentRunner: ...


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ConfirmRequest(BaseModel):
    approved: bool


class Conversation(BaseModel):
    conversation_id: str
    messages: list[ChatMessage]
    pending: AgentReply | None


UserDep = Annotated[UserContext, Depends(current_user)]
ConversationId = Annotated[
    str, Path(pattern=CONVERSATION_ID_PATTERN, description="Chosen by the client.")
]


def agent_runner(request: Request, session: SessionDep, user: UserDep) -> AgentRunner:
    agents: RunnerFactory | None = request.app.state.agents
    if agents is None:
        raise ConfigurationError("The assistant is not configured.")
    return agents.runner(session, user)


def rate_limit(request: Request, user: UserDep) -> None:
    limiter: RateLimiter = request.app.state.chat_limiter
    limiter.check(user.user_id)


RunnerDep = Annotated[AgentRunner, Depends(agent_runner)]


@router.post(
    "/{conversation_id}/messages", response_model=AgentReply, dependencies=[Depends(rate_limit)]
)
def send_message(
    conversation_id: ConversationId, body: MessageRequest, user: UserDep, runner: RunnerDep
) -> AgentReply:
    return runner.ask(thread_id(user, conversation_id), body.message)


@router.post(
    "/{conversation_id}/confirm", response_model=AgentReply, dependencies=[Depends(rate_limit)]
)
def confirm_action(
    conversation_id: ConversationId, body: ConfirmRequest, user: UserDep, runner: RunnerDep
) -> AgentReply:
    return runner.resume(thread_id(user, conversation_id), approved=body.approved)


@router.get("/{conversation_id}", response_model=Conversation)
def get_conversation(
    conversation_id: ConversationId, user: UserDep, runner: RunnerDep
) -> Conversation:
    thread = thread_id(user, conversation_id)
    return Conversation(
        conversation_id=conversation_id,
        messages=runner.history(thread),
        pending=runner.pending(thread),
    )
