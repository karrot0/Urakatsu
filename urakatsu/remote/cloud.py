"""Urakatsu Cloud: hosted detection, OCR, inpainting and translation at urakatsu.com.

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import httpx

from urakatsu.core.types import Availability

log = logging.getLogger(__name__)

URL = "https://api.urakatsu.com"
KEYS_URL = "https://urakatsu.com/dashboard/keys"
URL_ENV = "URAKATSU_CLOUD_URL"
KEY_ENV = "URAKATSU_CLOUD_API_KEY"
PROVIDER = "urakatsu-cloud"
HOSTED_TTL = 60.0
INFO_TIMEOUT = 3.0
ACCESS_TTL = 60.0

_hosted: Optional[tuple[float, str, Optional[frozenset[str]]]] = None
_hosted_lock = threading.Lock()
_access: dict[str, tuple[float, Availability]] = {}


def _saved() -> dict:
    """The Urakatsu Cloud provider's saved settings (its key and address)."""
    from urakatsu.core.config import load_config

    try:
        return load_config().providers.get(PROVIDER, {})
    except Exception as error:  # noqa: BLE001 - an unreadable config leaves the environment
        log.debug("could not read the Urakatsu Cloud settings: %s", error)
        return {}


def url(settings: Optional[dict] = None) -> str:
    """The cloud's address: ``server_url`` in the settings, else ``URAKATSU_CLOUD_URL``, else ``URL``."""
    settings = _saved() if settings is None else settings
    return (settings.get("server_url") or os.environ.get(URL_ENV) or URL).strip().rstrip("/")


def api_key(settings: Optional[dict] = None) -> Optional[str]:
    """The account key: ``settings`` (the provider's), else the saved configuration, else the environment."""
    settings = _saved() if settings is None else settings
    value = settings.get("api_key") or os.environ.get(KEY_ENV) or ""
    return value.strip() or None


def hosted() -> Optional[frozenset[str]]:
    global _hosted
    address = url()
    with _hosted_lock:
        if _hosted is not None and _hosted[1] == address and time.monotonic() - _hosted[0] < HOSTED_TTL:
            return _hosted[2]
        try:
            response = httpx.get(f"{address}/v1/info", timeout=INFO_TIMEOUT)
            response.raise_for_status()
            ids: Optional[frozenset[str]] = frozenset(str(p["id"]) for p in response.json().get("providers", []) if p.get("online", True))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            log.info("could not list the models Urakatsu Cloud at %s hosts: %s", address, error)
            ids = None
        # A failed listing is retried sooner than a good one is refreshed.
        _hosted = (time.monotonic() - (HOSTED_TTL - 60.0 if ids is None else 0.0), address, ids)
        return ids


def availability(provider: str) -> Availability:
    if api_key() is None:
        return Availability.no("add your Urakatsu Cloud key in Settings, Providers")
    ids = hosted()
    if ids is not None and provider not in ids:
        return Availability.no("not running on Urakatsu Cloud right now")
    return Availability.yes()


def image_access() -> Availability:
    """Whether the key may generate an image now: its image permission, and quota or credits left."""
    key = api_key()
    if key is None:
        return Availability.no("add your Urakatsu Cloud key in Settings, Providers")
    cached = _access.get(key)
    if cached is not None and time.monotonic() - cached[0] < ACCESS_TTL:
        return cached[1]
    try:
        response = httpx.get(f"{url()}/v1/account", headers={"Authorization": f"Bearer {key}"}, timeout=INFO_TIMEOUT)
        if response.status_code == 401:
            result = Availability.no(f"Urakatsu Cloud rejected the key; make a new one at {KEYS_URL}")
        else:
            response.raise_for_status()
            images = response.json()["imageGeneration"]
            result = Availability.yes() if images["available"] else Availability.no(str(images.get("reason") or "image generation is not available"))
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        result = Availability.no(f"cannot reach Urakatsu Cloud: {error}")
    _access[key] = (time.monotonic(), result)
    return result


def reset() -> None:
    """Forgets the hosted list and the image access answers (tests, and after the key changes)."""
    global _hosted
    with _hosted_lock:
        _hosted = None
    _access.clear()
