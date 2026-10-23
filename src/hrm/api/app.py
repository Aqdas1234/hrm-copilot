"""FastAPI app: health, policy RAG (/rag), HR data (hr_routes) and the agent chat (/chat).

create_app() is a factory. Without arguments it builds the real services once at startup
(RAG models and index, LLM clients, the agent and its conversation store); tests pass
fakes instead.

Every request gets an id (the X-Request-ID header, or a new one). It is bound to the
logging context, so every log line of that request carries it, and it is returned in the
X-Request-ID response header so a client can quote it when reporting a problem.
"""

import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import ExitStack, asynccontextmanager
from typing import Protocol

import groq
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from hrm import __version__
from hrm.agents.factory import AgentFactory
from hrm.agents.guardrails import check_input, check_output
from hrm.agents.memory import ConversationStore, open_memory
from hrm.agents.runner import NothingToConfirmError
from hrm.api import chat_routes, hr_routes
from hrm.api.chat_routes import RunnerFactory
from hrm.api.hr_routes import AuthConfig
from hrm.api.rate_limit import RateLimitedError, RateLimiter
from hrm.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ErrorResponse,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Timings,
)
from hrm.auth.access import AccessDeniedError
from hrm.auth.login import InvalidCredentialsError
from hrm.auth.tokens import InvalidTokenError
from hrm.core.config import get_settings
from hrm.core.errors import ConfigurationError, HRMError, LLMError
from hrm.core.logging import configure_logging, get_logger
from hrm.db.session import make_engine
from hrm.rag.models import Category, RetrievedChunk
from hrm.rag.service import AnswerResult, RAGService

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
QUOTE_CHARS = 400
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class PolicyService(Protocol):
    """What the API needs from the RAG service (RAGService, or a fake in tests)."""

    def search(
        self, query: str, k: int | None = None, *, category: Category | None = None
    ) -> list[RetrievedChunk]: ...

    def answer(self, question: str, *, category: Category | None = None) -> AnswerResult: ...


def create_app(
    service: PolicyService | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
    auth: AuthConfig | None = None,
    agents: RunnerFactory | None = None,
    conversations: ConversationStore | None = None,
    chat_limiter: RateLimiter | None = None,
) -> FastAPI:
    """Without arguments, everything is built from settings at startup; tests pass fakes.
    With a fake `service` and no `agents`, the chat endpoints answer 503."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        if service is None:
            configure_logging(settings.log_level, settings.log_format)
            log.info("api.starting", version=__version__)
        app.state.service = service or RAGService.from_settings(settings)  # loads models once
        app.state.session_factory = session_factory or sessionmaker(
            make_engine(settings.database_url.get_secret_value())
        )
        app.state.auth = auth or AuthConfig(
            secret=settings.jwt_secret.get_secret_value() if settings.jwt_secret else None,
            ttl_minutes=settings.jwt_ttl_minutes,
        )
        app.state.chat_limiter = chat_limiter or RateLimiter(settings.chat_messages_per_minute)
        with ExitStack() as stack:
            app.state.agents = agents
            app.state.conversations = conversations
            if agents is None and service is None:
                memory = stack.enter_context(open_memory(settings.memory_db_path))
                app.state.agents = AgentFactory.from_settings(
                    settings, memory, policies=app.state.service
                )
                app.state.conversations = ConversationStore(memory)
            yield

    app = FastAPI(
        title="HRM Copilot API",
        version=__version__,
        lifespan=lifespan,
        responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - start) * 1000),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()

    # Expected failures become clear HTTP errors; details stay in the logs, not the response.
    @app.exception_handler(LLMError)
    async def llm_error(request: Request, exc: LLMError) -> JSONResponse:
        log.error("api.llm_error", error=str(exc))
        return JSONResponse(
            status_code=502,
            content={"detail": "The answer service is temporarily unavailable. Please retry."},
        )

    @app.exception_handler(groq.APIError)
    async def model_api_error(request: Request, exc: groq.APIError) -> JSONResponse:
        # The agent's model calls (rate limit, outage, timeout). Same answer as LLMError.
        log.error("api.model_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=502,
            content={"detail": "The answer service is temporarily unavailable. Please retry."},
        )

    @app.exception_handler(RateLimitedError)
    async def rate_limited(request: Request, exc: RateLimitedError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    @app.exception_handler(NothingToConfirmError)
    async def nothing_to_confirm(request: Request, exc: NothingToConfirmError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        log.error("api.config_error", error=str(exc))
        return JSONResponse(status_code=503, content={"detail": "The service is not configured."})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the submitted values (e.g. a rejected password).
        # Keep where and why the request was invalid, never what was sent.
        errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(InvalidTokenError)
    @app.exception_handler(InvalidCredentialsError)
    async def unauthorized(request: Request, exc: HRMError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(AccessDeniedError)
    async def forbidden(request: Request, exc: AccessDeniedError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(HRMError)
    async def hrm_error(request: Request, exc: HRMError) -> JSONResponse:
        log.error("api.error", error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal error."})

    app.include_router(hr_routes.router)
    app.include_router(chat_routes.router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    # Plain `def` endpoints: FastAPI runs them in a thread pool, which suits the blocking
    # model and LLM calls inside the service.
    @app.post("/rag/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        service: PolicyService = request.app.state.service
        results = service.search(body.query, body.k, category=body.category)
        return SearchResponse(
            results=[
                SearchHit(
                    rank=r.rank,
                    score=r.score,
                    chunk_id=r.chunk.chunk_id,
                    doc_id=r.chunk.doc_id,
                    section=" > ".join(r.chunk.section_path),
                    text=r.chunk.text,
                )
                for r in results
            ]
        )

    @app.post("/rag/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        service: PolicyService = request.app.state.service
        input_check = check_input(body.question)
        if not input_check.ok:
            log.warning("rag.input_blocked", reason=input_check.blocked)
            return JSONResponse(  # type: ignore[return-value]
                status_code=400, content={"detail": input_check.refusal}
            )
        result = service.answer(input_check.text, category=body.category)
        output_check = check_output(result.answer.answer)
        if output_check.changed:
            log.warning("rag.output_redacted", findings=output_check.findings)
        texts = {r.chunk.chunk_id: r.chunk.text for r in result.context}
        return AskResponse(
            answer=output_check.text,
            found=result.answer.found,
            citations=[
                CitationOut(
                    number=c.number,
                    doc_id=c.doc_id,
                    section=" > ".join(c.section_path),
                    chunk_id=c.chunk_id,
                    quote=_quote(texts.get(c.chunk_id, "")),
                )
                for c in result.answer.citations
            ],
            timings=Timings(
                retrieve_ms=result.timings.retrieve_ms,
                generate_ms=result.timings.generate_ms,
                total_ms=result.timings.total_ms,
            ),
        )

    return app


def _quote(text: str, limit: int = QUOTE_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + " ..."
