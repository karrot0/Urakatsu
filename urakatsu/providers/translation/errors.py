"""Translation errors.

Messages keep the upstream ``Display`` text. All derive from ``ProcessingError`` so the
pipeline treats them as a failed stage run rather than a crash.
"""

from __future__ import annotations

from typing import Optional

from urakatsu.core.errors import ProcessingError


class TranslationError(ProcessingError):
    pass


class UnsupportedLanguage(TranslationError):
    def __init__(self, provider: str, language: str):
        super().__init__(f"{provider} does not support target language {language}")
        self.provider = provider
        self.language = language


class UnsupportedSourceLanguage(TranslationError):
    def __init__(self, provider: str, language: str):
        super().__init__(f"{provider} does not support source language {language}")
        self.provider = provider
        self.language = language


class SegmentCountError(TranslationError):
    def __init__(self, provider: str, expected: int, actual: int):
        super().__init__(f"{provider} returned {actual} segments; expected {expected}")
        self.provider = provider
        self.expected = expected
        self.actual = actual


class QuotaExceeded(TranslationError):
    """``status`` and ``body`` (when the service answered) tell a passing rate limit from
    an exhausted quota; the agent retries only the first."""

    def __init__(self, provider: str, status: Optional[int] = None, body: str = ""):
        super().__init__(f"{provider} quota or rate limit was exceeded")
        self.provider = provider
        self.status = status
        self.body = body


class ApiError(TranslationError):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(f"{provider} API request failed with HTTP {status}: {message}")
        self.provider = provider
        self.status = status
        self.message = message
