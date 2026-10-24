"""API key rotation across hosted translators (remote/base.py ``KeyRing``)."""

import json

import httpx
import pytest

from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.errors import ApiError
from urakatsu.providers.translation.language import Language
from urakatsu.providers.translation.remote import base
from urakatsu.providers.translation.remote.base import credential, credentials
from urakatsu.providers.translation.remote.lm_studio import LmStudioTranslator
from urakatsu.providers.translation.remote.openai import OpenAiTranslator


def build(cls, handler, api_key):
    provider = cls({"api_key": api_key, "translation": {"model": {"provider": cls.id, "model": "m"}}})
    provider.transport = httpx.MockTransport(handler)
    return provider


def request():
    return TranslationRequest(segments=["one"], target_language=Language.ENGLISH)


def ok():
    content = json.dumps({"translations": [{"id": 0, "text": "uno"}]})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def keyed(responses):
    """A handler that answers by bearer key and records the keys it saw."""
    seen = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        key = http_request.headers.get("Authorization", "").removeprefix("Bearer ")
        seen.append(key)
        return responses.get(key, ok)()

    return seen, handler


@pytest.fixture(autouse=True)
def fresh_rings(monkeypatch):
    monkeypatch.setattr(base, "_rings", {})
    for name in ("OPENAI_API_KEY", "LM_STUDIO_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_credentials_split_and_dedupe(monkeypatch):
    assert credentials({"api_key": " a, b\nc;a "}, "api_key", "UNSET") == ["a", "b", "c"]
    assert credentials({"api_key": " "}, "api_key", "UNSET") == []
    assert credential({"api_key": "a,b"}, "api_key", "UNSET") == "a"
    monkeypatch.setenv("OPENAI_API_KEY", "x y")
    assert credentials({}, "api_key", "OPENAI_API_KEY") == ["x", "y"]


async def test_single_key_is_used_directly():
    seen, handler = keyed({})
    provider = build(OpenAiTranslator, handler, "only")
    assert await provider.translate(request()) == ["uno"]
    assert seen == ["only"]


async def test_requests_take_turns():
    seen, handler = keyed({})
    provider = build(OpenAiTranslator, handler, "a,b,c")
    for _ in range(4):
        await provider.translate(request())
    assert seen == ["a", "b", "c", "a"]


async def test_rotation_carries_across_instances():
    seen, handler = keyed({})
    await build(OpenAiTranslator, handler, "a,b").translate(request())
    await build(OpenAiTranslator, handler, "a,b").translate(request())
    assert seen == ["a", "b"]


async def test_quota_moves_on_and_rests_the_key():
    seen, handler = keyed({"a": lambda: httpx.Response(429, text="slow down")})
    provider = build(OpenAiTranslator, handler, "a,b")
    assert await provider.translate(request()) == ["uno"]
    # The next turn starts at b anyway; the one after would start at a, which is resting.
    await provider.translate(request())
    await provider.translate(request())
    assert seen == ["a", "b", "b", "b"]


async def test_rejected_key_moves_on():
    seen, handler = keyed({"a": lambda: httpx.Response(401, text="bad key")})
    provider = build(OpenAiTranslator, handler, "a,b")
    assert await provider.translate(request()) == ["uno"]
    assert seen == ["a", "b"]
    assert "a" in base.key_ring(provider.id).resting


async def test_other_errors_do_not_rotate():
    seen, handler = keyed({"a": lambda: httpx.Response(500, text="boom")})
    provider = build(OpenAiTranslator, handler, "a,b")
    with pytest.raises(ApiError) as failure:
        await provider.translate(request())
    assert failure.value.status == 500
    assert seen == ["a"]


async def test_every_key_failing_raises_the_last_error():
    seen, handler = keyed(
        {
            "a": lambda: httpx.Response(429, text="slow down"),
            "b": lambda: httpx.Response(403, text="forbidden"),
        }
    )
    provider = build(OpenAiTranslator, handler, "a,b")
    with pytest.raises(ApiError) as failure:
        await provider.translate(request())
    assert failure.value.status == 403
    # Resting keys are still tried, soonest-ready first (a rests 60 s, b an hour), even
    # though this turn would otherwise start at b.
    with pytest.raises(ApiError):
        await provider.translate(request())
    assert seen == ["a", "b", "a", "b"]


async def test_success_ends_the_rest():
    responses = {"a": lambda: httpx.Response(429, text="slow down")}
    seen, handler = keyed(responses)
    provider = build(OpenAiTranslator, handler, "a,b")
    await provider.translate(request())
    responses.clear()
    ring = base.key_ring(provider.id)
    ring.resting["b"] = ring.resting["a"] + 1  # b rests longer, so a is tried first
    await provider.translate(request())
    assert seen[-1] == "a"
    assert "a" not in ring.resting


async def test_optional_keys_rotate_too():
    seen, handler = keyed({})
    provider = build(LmStudioTranslator, handler, "a,b")
    await provider.translate(request())
    await provider.translate(request())
    assert seen == ["a", "b"]
