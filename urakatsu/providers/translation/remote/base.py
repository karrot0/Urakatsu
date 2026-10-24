"""Shared plumbing for hosted translation providers.

- ``send_json`` / ``send_json_sync``: one request, the upstream response policy. A non-2xx
  reply becomes ``QuotaExceeded`` when the status is 429 or the body names a quota
  (``insufficient_quota``, ``resource_exhausted``, ``rate limit exceeded``,
  ``credit balance is too low``); otherwise ``ApiError`` with the first 4096 characters
  of the body. A 2xx body that is not JSON is a decode error.
- Listing a server on this machine (LM Studio, a local OpenAI-compatible server) connects
  with a ``LOOPBACK_CONNECT_TIMEOUT`` timeout. A local server accepts at once; one that is
  not running would otherwise take seconds to refuse on Windows (about two seconds per
  address ``localhost`` resolves to), and every model list waits for it.
- API keys: Upstream reads them from the OS credential store. Here each provider takes
  ``api_key`` from its settings, falling back to an environment variable named by the
  provider (``OPENAI_API_KEY`` and so on). A missing key is an error for translation
  and an empty model list for listing.
- Key rotation (not in upstream, which holds one key per provider): the setting or the
  environment variable may list several keys separated by commas or whitespace. Each
  translation starts at the next key in turn; a key that hits a quota or rate limit
  rests for ``QUOTA_COOLDOWN`` seconds and one the service rejects (HTTP 401/403) for
  ``REJECTED_COOLDOWN``, and the request moves on to the next key. Resting keys are
  tried last rather than skipped, so a request fails only once every key has failed,
  with the last key's error. Model listing uses the first key.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from contextvars import ContextVar
from typing import Any, ClassVar, Mapping, Optional

import httpx

from urakatsu.core.config import GenerationConfig

from ..backend import TranslationRequest
from ..base import TranslatorProvider
from ..errors import ApiError, QuotaExceeded, TranslationError

TRANSLATE_TIMEOUT = httpx.Timeout(30.0, read=180.0)
LISTING_TIMEOUT = 5.0
LOOPBACK_CONNECT_TIMEOUT = 0.5
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
ERROR_BODY_LIMIT = 4096
QUOTA_MARKERS = (
    "insufficient_quota",
    "resource_exhausted",
    "rate limit exceeded",
    "credit balance is too low",
)
KEY_SEPARATORS = re.compile(r"[\s,;]+")
QUOTA_COOLDOWN = 60.0
REJECTED_COOLDOWN = 3600.0
REJECTED_STATUSES = (401, 403)


def credentials(settings: Mapping[str, Any], key: str, env: str) -> list[str]:
    """Every configured key, in order, or those in the environment variable ``env`` when unset."""
    value = settings.get(key) or os.environ.get(env)
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(part for part in KEY_SEPARATORS.split(value) if part))


def credential(settings: Mapping[str, Any], key: str, env: str) -> Optional[str]:
    """The first configured key, or ``None``."""
    keys = credentials(settings, key, env)
    return keys[0] if keys else None


class KeyRing:
    """One provider's rotation state: the next starting key and the keys resting after a failure."""

    def __init__(self) -> None:
        self.turn = 0
        self.resting: dict[str, float] = {}

    def order(self, keys: list[str]) -> list[str]:
        """The keys to try for one request: ready keys from the next turn, then resting keys."""
        start = self.turn % len(keys)
        self.turn += 1
        rotated = keys[start:] + keys[:start]
        now = time.monotonic()
        ready = [key for key in rotated if self.resting.get(key, 0.0) <= now]
        resting = sorted((key for key in rotated if key not in ready), key=self.resting.__getitem__)
        return ready + resting

    def rest(self, key: str, seconds: float) -> None:
        self.resting[key] = time.monotonic() + seconds

    def recover(self, key: str) -> None:
        self.resting.pop(key, None)


# Process-wide, so rotation carries across provider instances and pipeline runs.
_rings: dict[str, KeyRing] = {}
# The key the current request is using while ``RemoteTranslator`` rotates.
_active_key: ContextVar[Optional[str]] = ContextVar("urakatsu_active_api_key", default=None)


def key_ring(provider: str) -> KeyRing:
    return _rings.setdefault(provider, KeyRing())


def decode_response(provider: str, status: int, text: str) -> Any:
    """``send_json``'s response policy, shared by the async and sync paths."""
    if not 200 <= status < 300:
        if status == 429 or any(marker in text.lower() for marker in QUOTA_MARKERS):
            raise QuotaExceeded(provider, status, text[:ERROR_BODY_LIMIT])
        raise ApiError(provider, status, text[:ERROR_BODY_LIMIT])
    try:
        return json.loads(text)
    except ValueError as error:
        raise TranslationError(f"failed to decode {provider} response: {error}") from error


def field(provider: str, value: Any, key: str) -> Any:
    """A required member of a decoded response, failing like serde does."""
    if not isinstance(value, dict) or key not in value:
        raise TranslationError(f"failed to decode {provider} response: missing field `{key}`")
    return value[key]


async def send_json(
    provider: str,
    method: str,
    url: str,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, str]] = None,
    json_body: Any = None,
    form: Any = None,
) -> Any:
    body = _form_body(form, headers)
    async with httpx.AsyncClient(transport=transport, timeout=TRANSLATE_TIMEOUT) as client:
        try:
            if form is not None:
                response = await client.request(method, url, headers=body[1], params=params, content=body[0])
            else:
                response = await client.request(method, url, headers=headers, params=params, json=json_body)
        except httpx.HTTPError as error:
            raise TranslationError(f"{provider} request failed: {error}") from error
    return decode_response(provider, response.status_code, response.text)


def _form_body(form: Any, headers: Optional[Mapping[str, str]]) -> tuple[bytes, dict[str, str]]:
    """``application/x-www-form-urlencoded`` body (a list of pairs keeps repeated keys, like reqwest's ``.form``)."""
    merged = dict(headers or {})
    merged["Content-Type"] = "application/x-www-form-urlencoded"
    return urllib.parse.urlencode(form or []).encode("ascii"), merged


def send_json_sync(
    provider: str,
    method: str,
    url: str,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, str]] = None,
) -> Any:
    timeout = httpx.Timeout(LISTING_TIMEOUT)
    if (urllib.parse.urlsplit(url).hostname or "").lower() in LOOPBACK_HOSTS:
        timeout = httpx.Timeout(LISTING_TIMEOUT, connect=LOOPBACK_CONNECT_TIMEOUT)
    with httpx.Client(transport=transport, timeout=timeout) as client:
        try:
            response = client.request(method, url, headers=headers, params=params)
        except httpx.HTTPError as error:
            raise TranslationError(f"{provider} request failed: {error}") from error
    return decode_response(provider, response.status_code, response.text)


class RemoteTranslator(TranslatorProvider):
    """A hosted translation service. Subclasses implement ``_translate`` and ``list_models``."""

    runtimes: ClassVar[tuple[str, ...]] = ()
    # Environment variable holding the API key when settings do not set one.
    env_key: ClassVar[Optional[str]] = None

    def api_keys(self) -> list[str]:
        return credentials(self.settings.model_dump(), "api_key", self.env_key or "")

    def optional_api_key(self) -> Optional[str]:
        """The key for the current request: the rotation's pick, else the first configured key."""
        active = _active_key.get()
        if active is not None:
            return active
        keys = self.api_keys()
        return keys[0] if keys else None

    def api_key(self) -> str:
        key = self.optional_api_key()
        if key is None:
            raise TranslationError(f"{self.id} API key is not configured")
        return key

    async def _dispatch(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        keys = self.api_keys()
        if len(keys) < 2:
            return await self._translate(request, generation)
        ring = key_ring(self.id)
        failure: Optional[TranslationError] = None
        for key in ring.order(keys):
            token = _active_key.set(key)
            try:
                translated = await self._translate(request, generation)
            except QuotaExceeded as error:
                ring.rest(key, QUOTA_COOLDOWN)
                failure = error
            except ApiError as error:
                if error.status not in REJECTED_STATUSES:
                    raise
                ring.rest(key, REJECTED_COOLDOWN)
                failure = error
            else:
                ring.recover(key)
                return translated
            finally:
                _active_key.reset(token)
        assert failure is not None
        raise failure

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await send_json(self.id, "POST", url, transport=self.transport, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await send_json(self.id, "GET", url, transport=self.transport, **kwargs)
