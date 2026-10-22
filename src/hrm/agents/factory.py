"""Builds the agent for a logged-in user: the one place that wires everything together.

Two lifetimes:
- per process (AgentFactory): the chat model, policy search (loads local models once),
  the HTTP client, the payroll client (its circuit breaker must remember failures across
  requests) and the conversation store;
- per request (runner()): the database session and everything bound to the user,
  i.e. the tools, the system prompt and the compiled graph.
"""

from dataclasses import dataclass
from datetime import date

import httpx
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session

from hrm.agents.graph import AgentLimits, build_agent
from hrm.agents.prompts import system_prompt
from hrm.agents.router import build_router_agent
from hrm.agents.runner import AgentRunner
from hrm.agents.tools import ToolDeps, build_tools
from hrm.auth.context import UserContext
from hrm.core.config import Settings
from hrm.core.llm import get_fast_model
from hrm.db import repository as repo
from hrm.integrations.currency import CurrencyService
from hrm.integrations.holidays import HolidayService
from hrm.integrations.http import HTTPGetter
from hrm.integrations.payroll import PayrollClient
from hrm.rag.service import RAGService
from hrm.rag.tool import PolicySearch


@dataclass(frozen=True)
class AgentFactory:
    settings: Settings
    llm: BaseChatModel
    policies: PolicySearch
    http: HTTPGetter
    payroll: PayrollClient
    checkpointer: BaseCheckpointSaver[str]

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        checkpointer: BaseCheckpointSaver[str],
        *,
        policies: PolicySearch | None = None,
    ) -> "AgentFactory":
        """`policies`: an already loaded RAGService (the API shares its own), else a new one."""
        http = httpx.Client(timeout=settings.http_timeout_s)
        return cls(
            settings=settings,
            # The small model decides which tools to call; policy answers inside
            # search_hr_policies come from retrieval. B7 measures whether this is enough.
            llm=get_fast_model(settings),
            policies=policies or RAGService.from_settings(settings),
            http=http,
            payroll=PayrollClient(
                http, base_url=settings.payroll_api_url, attempts=settings.http_retries
            ),
            checkpointer=checkpointer,
        )

    def runner(self, session: Session, user: UserContext, today: date | None = None) -> AgentRunner:
        today = today or date.today()
        employee = repo.get_employee(session, user.employee_id)
        if employee is None:  # a user always has an employee record (foreign key)
            raise LookupError(f"No employee record for user {user.user_id}")
        retries = self.settings.http_retries
        deps = ToolDeps(
            session=session,
            user=user,
            policies=self.policies,
            holidays=HolidayService(
                session, self.http, base_url=self.settings.holidays_api_url, attempts=retries
            ),
            rates=CurrencyService(
                session, self.http, base_url=self.settings.currency_api_url, attempts=retries
            ),
            payroll=self.payroll,
            today=today,
        )
        tools = build_tools(deps)
        prompt = system_prompt(name=employee.full_name, role=user.role, today=today)
        limits = AgentLimits.from_settings(self.settings)
        if self.settings.agent_mode == "router":
            agent = build_router_agent(
                self.llm, self.llm, tools, prompt, limits, checkpointer=self.checkpointer
            )
        else:
            agent = build_agent(self.llm, tools, prompt, limits, checkpointer=self.checkpointer)
        return AgentRunner(agent)
