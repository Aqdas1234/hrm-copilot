"""Project exception types.

Everything we raise on purpose derives from HRMError, so callers (CLI, API) can tell
our own, expected failures apart from bugs and show a clean message for them.
"""


class HRMError(Exception):
    """Base class for all errors raised deliberately by this project."""


class ConfigurationError(HRMError):
    """A required setting is missing or invalid (for example, no API key)."""


class DocumentLoadError(HRMError):
    """A policy document or the catalog is missing, invalid or out of sync."""


class LLMError(HRMError):
    """The language model call failed after retries, or returned something unusable."""
