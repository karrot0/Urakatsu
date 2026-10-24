"""The chat backend interface and the streaming plumbing the wire formats share.

``failure_kind`` sorts a failed model step for the loop (``session.py``): ``transient``
(a dropped connection, a timeout, HTTP 408/409/425/5xx, a 429 rate limit that is not an
exhausted quota, or a provider's own overloaded/server error event) is worth sending
again; ``overflow`` (the request was longer than the model's context window) is worth
sending again shorter. The decision reads the HTTP status, not the message text, except
where only the body can tell (a 429 rate limit from a spent quota, a 400 that is a context
overflow).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal, Mapping, Optional

import httpx

from urakatsu.providers.translation.errors import ApiError, QuotaExceeded, TranslationError
from urakatsu.providers.translation.remote.base import decode_response

from ..messages import BackendEvent, Message, ToolSpec

STREAM_TIMEOUT = httpx.Timeout(30.0, read=300.0)
STREAM_IDLE = 300.0
TRANSIENT_STATUSES = frozenset({408, 409, 425, 500, 502, 503, 504, 529})
# A 429 whose body says one of these will not pass by waiting.
SPENT_QUOTA_MARKERS = ("insufficient_quota", "exceeded your current quota", "credit balance is too low", "billing", "usage_limit_reached")
# A 400/413 whose body says one of these was too long for the model's context window.
OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "context size",
    "exceed_context_size",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "input token count",
)

FailureKind = Literal["transient", "overflow"]


class AgentError(TranslationError):
    """A chat backend failed; the message is shown in the conversation.

    ``transient`` and ``overflow`` mark failures the loop may answer by sending the step
    again (see ``failure_kind``)."""

    def __init__(self, message: str, transient: bool = False, overflow: bool = False):
        super().__init__(message)
        self.transient = transient
        self.overflow = overflow


def overflowed(text: str) -> bool:
    text = text.lower()
    return any(marker in text for marker in OVERFLOW_MARKERS)


def failure_kind(error: BaseException) -> Optional[FailureKind]:
    """Whether a failed model step may succeed if sent again (``transient``), if sent shorter
    (``overflow``), or not at all (``None``)."""
    if isinstance(error, AgentError):
        return "overflow" if error.overflow else "transient" if error.transient else None
    if isinstance(error, QuotaExceeded):
        spent = any(marker in error.body.lower() for marker in SPENT_QUOTA_MARKERS)
        return "transient" if error.status == 429 and not spent else None
    if isinstance(error, ApiError):
        if error.status in (400, 413) and overflowed(error.message):
            return "overflow"
        return "transient" if error.status in TRANSIENT_STATUSES else None
    return None


class ChatBackend(ABC):
    """One model behind one wire format. ``stream`` runs a single model step."""

    # ``provider:model``, as stored on the assistant messages it writes.
    ref: str = ""
    transport: Optional[httpx.AsyncBaseTransport] = None
    # The model's context window in tokens, when the backend knows it; the loop assumes
    # ``compaction.DEFAULT_CONTEXT_WINDOW`` otherwise.
    context_window: Optional[int] = None

    @abstractmethod
    def stream(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[BackendEvent]:
        """Text and reasoning deltas as they arrive, then the complete tool calls the model asked for."""


async def sse_events(
    provider: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Any,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> AsyncIterator[tuple[str, Any]]:
    """POSTs ``body`` and yields ``(event, data)`` for each server-sent event; ``[DONE]`` ends the stream.

    A non-2xx answer fails like a translation request does (quota errors become
    ``QuotaExceeded``), so API key rotation and error text match the translators.
    """
    async with httpx.AsyncClient(transport=transport, timeout=STREAM_TIMEOUT) as client:
        try:
            async with client.stream("POST", url, headers=dict(headers), json=body) as response:
                if not 200 <= response.status_code < 300:
                    text = (await response.aread()).decode("utf-8", "replace")
                    decode_response(provider, response.status_code, text)
                event = "message"
                data: list[str] = []
                last = time.monotonic()
                async for line in response.aiter_lines():
                    if line.startswith(":"):
                        if time.monotonic() - last > STREAM_IDLE:
                            raise AgentError(f"{provider} sent nothing but keep-alives for {STREAM_IDLE:.0f} seconds", transient=True)
                        continue
                    if line:  # not the blank line that ends each keep-alive too
                        last = time.monotonic()
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        payload = "\n".join(data)
                        data = []
                        if payload.strip() == "[DONE]":
                            return
                        try:
                            yield event, json.loads(payload)
                        except ValueError as error:
                            raise AgentError(f"{provider} sent an unreadable event: {payload[:200]}") from error
                        event = "message"
                if data and "\n".join(data).strip() != "[DONE]":
                    yield event, json.loads("\n".join(data))
        except httpx.HTTPError as error:
            # Connection and timeout failures may pass; a malformed request will not.
            raise AgentError(f"{provider} request failed: {error}", transient=isinstance(error, httpx.TransportError)) from error
