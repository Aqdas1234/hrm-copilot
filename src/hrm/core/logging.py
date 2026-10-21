"""Structured logging with structlog.

Log calls take an event name plus key/value fields, e.g.
    log.info("retrieval.done", query_id=qid, hits=5, ms=42)
which stays readable in the console and becomes one JSON object per line in "json" mode.
Logs go to stderr so they never mix with command output on stdout.
"""

import logging
import sys

import structlog

from hrm.core.config import LogFormat, LogLevel


def configure_logging(level: LogLevel = "INFO", fmt: LogFormat = "console") -> None:
    """Set up structlog (and the standard logging used by libraries) once at startup."""
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # e.g. a request_id bound per API request
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=max(logging.getLevelName(level), logging.WARNING), stream=sys.stderr)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a logger tagged with the module name. Use: log = get_logger(__name__)."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(module=name)
    return logger
