"""Claude, Gemini and Caiyun translators."""

import base64
import json

import httpx
import pytest
from PIL import Image

from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.errors import (
    ApiError,
    QuotaExceeded,
    SegmentCountError,
    TranslationError,
    UnsupportedLanguage,
)
from urakatsu.providers.translation.language import Language
from urakatsu.providers.translation.remote.caiyun import CaiyunTranslator
from urakatsu.providers.translation.remote.claude import ClaudeTranslator
from urakatsu.providers.translation.remote.gemini import GeminiTranslator


def build(cls, handler=None, *, model="m", generation=None, api_key="secret", provider_id=None):
    settings = {
        "translation": {
            "model": {"provider": provider_id or cls.id, "model": model},
            "generation": generation or {},
        },
    }
    if api_key is not None:
        settings["api_key"] = api_key
    provider = cls(settings)
    if handler is not None:
        provider.transport = httpx.MockTransport(handler)
    return provider


def request(segments=("one", "two"), **kwargs):
    kwargs.setdefault("target_language", Language.ENGLISH)
    return TranslationRequest(segments=list(segments), **kwargs)


def reply(payload, status=200):
    return httpx.Response(status, json=payload)


def translation_json(*texts):
    return json.dumps({"translations": [{"id": index, "text": text} for index, text in enumerate(texts)]})


def recorder(response):
    seen = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return response if isinstance(response, httpx.Response) else response(http_request)

    return seen, handler


@pytest.fixture(autouse=True)
def no_ambient_keys(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "CAIYUN_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# --- Claude -----------------------------------------------------------------------------------


def claude_ok(*texts):
    return reply({"content": [{"type": "thinking"}, {"type": "text", "text": translation_json(*texts)}]})


async def test_claude_request_shape_and_parsing():
    seen, handler = recorder(claude_ok("uno", "dos"))
    provider = build(
        ClaudeTranslator,
        handler,
        model="claude-opus-4-8",
        generation={"temperature": 0.2, "max_tokens": 321, "top_p": 0.9, "reasoning": True},
    )
    result = await provider.translate(request())
    assert result == ["uno", "dos"]
    sent = seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == "secret"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in sent.headers
    body = json.loads(sent.content)
    assert list(body) == ["model", "max_tokens", "system", "messages", "temperature", "thinking", "output_config"]
    assert body["model"] == "claude-opus-4-8"
    assert body["max_tokens"] == 321
    assert body["temperature"] == 0.2
    assert body["thinking"] == {"type": "adaptive"}
    assert "top_p" not in body
    assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    assert [block["type"] for block in body["messages"][0]["content"]] == ["text"]
    payload = json.loads(body["messages"][0]["content"][0]["text"])
    assert [segment["text"] for segment in payload["segments"]] == ["one", "two"]
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["translations"]["maxItems"] == 2


async def test_claude_defaults_omit_optional_fields_and_disable_thinking():
    seen, handler = recorder(claude_ok("a", "b"))
    provider = build(ClaudeTranslator, handler, generation={"reasoning": False})
    await provider.translate(request())
    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 8192
    assert "temperature" not in body
    assert body["thinking"] == {"type": "disabled"}

    seen.clear()
    provider = build(ClaudeTranslator, handler, generation={"reasoning": None})
    await provider.translate(request())
    assert "thinking" not in json.loads(seen[0].content)


async def test_claude_sends_base64_jpeg_image_block():
    seen, handler = recorder(claude_ok("a", "b"))
    provider = build(ClaudeTranslator, handler)
    await provider.translate(request(image=Image.new("RGB", (4, 4), "white")))
    content = json.loads(seen[0].content)["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "image"]
    source = content[1]["source"]
    assert source["type"] == "base64" and source["media_type"] == "image/jpeg"
    assert base64.b64decode(source["data"])[:2] == b"\xff\xd8"


async def test_claude_vision_disabled_drops_image():
    seen, handler = recorder(claude_ok("a", "b"))
    provider = build(ClaudeTranslator, handler, generation={"vision": False})
    await provider.translate(request(image=Image.new("RGB", (4, 4))))
    assert len(json.loads(seen[0].content)["messages"][0]["content"]) == 1


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (httpx.Response(429, text="slow down"), QuotaExceeded, "quota"),
        (httpx.Response(400, text="bad request"), ApiError, "HTTP 400"),
        (httpx.Response(400, text="Your credit balance is too low"), QuotaExceeded, "quota"),
    ],
)
async def test_claude_http_errors(response, error, match):
    provider = build(ClaudeTranslator, lambda _: response)
    with pytest.raises(error, match=match):
        await provider.translate(request())


async def test_claude_missing_ids_keep_source_text():
    # prompt.translations restores order and keeps the source for missing ids, so the count always matches.
    provider = build(ClaudeTranslator, lambda _: claude_ok("only"))
    assert await provider.translate(request()) == ["only", "two"]


async def test_claude_no_text_content_and_malformed_json():
    provider = build(ClaudeTranslator, lambda _: reply({"content": [{"type": "thinking"}]}))
    with pytest.raises(TranslationError, match="Claude returned no text content"):
        await provider.translate(request())
    provider = build(ClaudeTranslator, lambda _: reply({"content": [{"type": "text", "text": "not json"}]}))
    with pytest.raises(TranslationError, match="invalid translation JSON"):
        await provider.translate(request())


async def test_claude_missing_api_key():
    provider = build(ClaudeTranslator, lambda _: claude_ok("a", "b"), api_key=None)
    with pytest.raises(TranslationError, match="API key is not configured"):
        await provider.translate(request())


async def test_claude_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    seen, handler = recorder(claude_ok("a", "b"))
    await build(ClaudeTranslator, handler, api_key=None).translate(request())
    assert seen[0].headers["x-api-key"] == "from-env"


def test_claude_list_models():
    def models_reply(http_request):
        assert http_request.method == "GET"
        assert http_request.url.path == "/v1/models"
        assert http_request.url.params["limit"] == "1000"
        assert http_request.headers["x-api-key"] == "secret"
        assert http_request.headers["anthropic-version"] == "2023-06-01"
        return reply(
            {
                "data": [
                    {
                        "id": "claude-opus-4-8",
                        "display_name": "Claude Opus 4.8",
                        "capabilities": {
                            "image_input": {"supported": True},
                            "thinking": {"types": {"adaptive": {"supported": True}}},
                        },
                    },
                    {
                        "id": "claude-old",
                        "display_name": "Claude Old",
                        "capabilities": {
                            "image_input": {"supported": False},
                            "thinking": {"types": {"adaptive": {"supported": False}}},
                        },
                    },
                ]
            }
        )

    models = ClaudeTranslator.list_models({"api_key": "secret"}, transport=httpx.MockTransport(models_reply))
    assert [(m.provider, m.provider_name, m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("claude", "Claude", "claude-opus-4-8", "Claude Opus 4.8", True, True),
        ("claude", "Claude", "claude-old", "Claude Old", False, False),
    ]


def test_claude_list_models_without_key_is_empty_and_errors_propagate():
    assert ClaudeTranslator.list_models({}) == []
    with pytest.raises(ApiError):
        ClaudeTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(lambda _: httpx.Response(401, text="no")))


# --- Gemini -----------------------------------------------------------------------------------


def gemini_ok(*texts):
    return reply({"candidates": [{"content": {"parts": [{"text": translation_json(*texts)}]}}]})


async def test_gemini_request_shape_and_parsing():
    seen, handler = recorder(gemini_ok("uno", "dos"))
    provider = build(
        GeminiTranslator,
        handler,
        model="gemini-3.7-flash",
        generation={"temperature": 0.3, "max_tokens": 99, "top_p": 0.5, "reasoning": True},
    )
    assert await provider.translate(request()) == ["uno", "dos"]
    sent = seen[0]
    assert sent.method == "POST"
    assert sent.url.host == "generativelanguage.googleapis.com"
    assert sent.url.path == "/v1beta/models/gemini-3.7-flash:generateContent"
    assert sent.url.params["key"] == "secret"
    assert "authorization" not in sent.headers and "x-goog-api-key" not in sent.headers
    body = json.loads(sent.content)
    assert list(body) == ["system_instruction", "contents", "generation_config"]
    assert body["system_instruction"]["parts"][0]["text"]
    assert len(body["contents"]) == 1 and "role" not in body["contents"][0]
    assert [list(part) for part in body["contents"][0]["parts"]] == [["text"]]
    config = body["generation_config"]
    assert list(config) == [
        "temperature",
        "maxOutputTokens",
        "thinkingConfig",
        "responseMimeType",
        "responseJsonSchema",
    ]
    assert config["temperature"] == 0.3 and config["maxOutputTokens"] == 99
    assert config["thinkingConfig"] == {"thinkingLevel": "high"}
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["translations"]["maxItems"] == 2


async def test_gemini_omits_unset_options():
    seen, handler = recorder(gemini_ok("a", "b"))
    provider = build(GeminiTranslator, handler, generation={"reasoning": None})
    await provider.translate(request())
    config = json.loads(seen[0].content)["generation_config"]
    assert list(config) == ["responseMimeType", "responseJsonSchema"]


@pytest.mark.parametrize(
    ("model", "reasoning", "expected"),
    [
        ("gemini-2.5-flash", True, {"thinkingBudget": -1}),
        ("gemini-2.5-flash", False, {"thinkingBudget": 0}),
        ("gemini-2.5-pro", False, {"thinkingBudget": 128}),
        ("gemini-3.7-flash", False, {"thinkingLevel": "low"}),
        ("gemma-4-31b-it", False, {"thinkingLevel": "minimal"}),
        ("gemma-4-31b-it", True, {"thinkingLevel": "high"}),
    ],
)
async def test_gemini_thinking_controls(model, reasoning, expected):
    seen, handler = recorder(gemini_ok("a", "b"))
    provider = build(GeminiTranslator, handler, model=model, generation={"reasoning": reasoning})
    await provider.translate(request())
    assert json.loads(seen[0].content)["generation_config"]["thinkingConfig"] == expected


async def test_gemini_sends_inline_data_image():
    seen, handler = recorder(gemini_ok("a", "b"))
    provider = build(GeminiTranslator, handler)
    await provider.translate(request(image=Image.new("RGB", (4, 4), "white")))
    parts = json.loads(seen[0].content)["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(parts[1]["inlineData"]["data"])[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (httpx.Response(429, text="x"), QuotaExceeded, "quota"),
        (httpx.Response(400, text="bad"), ApiError, "HTTP 400"),
        (httpx.Response(403, text='{"status":"RESOURCE_EXHAUSTED"}'), QuotaExceeded, "quota"),
    ],
)
async def test_gemini_http_errors(response, error, match):
    provider = build(GeminiTranslator, lambda _: response)
    with pytest.raises(error, match=match):
        await provider.translate(request())


async def test_gemini_no_candidate_content():
    provider = build(GeminiTranslator, lambda _: reply({"candidates": []}))
    with pytest.raises(TranslationError, match="Gemini returned no candidate content"):
        await provider.translate(request())
    provider = build(GeminiTranslator, lambda _: reply({"candidates": [{"content": {"parts": []}}]}))
    with pytest.raises(TranslationError, match="Gemini returned no candidate content"):
        await provider.translate(request())
    provider = build(GeminiTranslator, lambda _: reply({"candidates": [{"content": {"parts": [{"inlineData": {}}]}}]}))
    with pytest.raises(TranslationError, match="missing field `text`"):
        await provider.translate(request())


async def test_gemini_missing_api_key():
    provider = build(GeminiTranslator, lambda _: gemini_ok("a", "b"), api_key=None)
    with pytest.raises(TranslationError, match="API key is not configured"):
        await provider.translate(request())


def test_gemini_list_models_filters_and_maps():
    def models_reply(http_request):
        assert http_request.method == "GET"
        assert http_request.url.path == "/v1beta/models"
        assert http_request.url.params["key"] == "secret"
        assert http_request.url.params["pageSize"] == "1000"
        return reply(
            {
                "models": [
                    {
                        "name": "models/gemini-3.7-flash",
                        "displayName": "Gemini 3.7 Flash",
                        "supportedGenerationMethods": ["generateContent"],
                        "thinking": True,
                    },
                    {
                        "name": "models/gemini-2.0-flash-lite",
                        "displayName": "Gemini 2.0 Flash-Lite",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/gemini-3.1-flash-image",
                        "displayName": "Image",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-3.1-flash-tts-preview",
                        "displayName": "TTS",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "displayName": "Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/aqa",
                        "displayName": "AQA",
                        "supportedGenerationMethods": ["generateAnswer"],
                    },
                ]
            }
        )

    models = GeminiTranslator.list_models({"api_key": "secret"}, transport=httpx.MockTransport(models_reply))
    assert [(m.provider, m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("gemini", "gemini-3.7-flash", "Gemini 3.7 Flash", True, True),
        ("gemini", "gemini-2.0-flash-lite", "Gemini 2.0 Flash-Lite", True, False),
    ]


def test_gemini_list_models_without_key_is_empty():
    assert GeminiTranslator.list_models({}) == []


# --- Caiyun -----------------------------------------------------------------------------------


async def test_caiyun_request_shape_and_parsing():
    seen, handler = recorder(reply({"rc": 0, "target": ["uno", "dos"]}))
    provider = build(CaiyunTranslator, handler, model=None)
    result = await provider.translate(request(target_language=Language.JAPANESE, instructions="ignored"))
    assert result == ["uno", "dos"]
    sent = seen[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.interpreter.caiyunai.com/v1/translator"
    assert sent.headers["x-authorization"] == "token secret"
    assert json.loads(sent.content) == {
        "source": ["one", "two"],
        "trans_type": "auto2ja",
        "request_id": "koharu-translator",
        "detect": True,
        "media": "text",
    }
    assert list(json.loads(sent.content)) == ["source", "trans_type", "request_id", "detect", "media"]


async def test_caiyun_single_string_target_and_rc_default():
    provider = build(CaiyunTranslator, lambda _: reply({"target": "solo"}), model=None)
    assert await provider.translate(request(("one",), target_language=Language.CHINESE_TRADITIONAL)) == ["solo"]


async def test_caiyun_never_sends_image():
    seen, handler = recorder(reply({"target": ["a", "b"]}))
    provider = build(CaiyunTranslator, handler, model=None)
    await provider.translate(request(image=Image.new("RGB", (4, 4))))
    assert "image" not in json.loads(seen[0].content)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"rc": 3, "msg": "bad token", "message": "m", "error": "e"}, "Caiyun returned rc=3: bad token"),
        ({"rc": 3, "message": "m", "error": "e"}, "Caiyun returned rc=3: m"),
        ({"rc": 3, "error": "e"}, "Caiyun returned rc=3: e"),
        ({"rc": -1}, "Caiyun returned rc=-1: unknown error"),
        ({"rc": 0}, "Caiyun returned no target"),
    ],
)
async def test_caiyun_response_failures(payload, message):
    provider = build(CaiyunTranslator, lambda _: reply(payload), model=None)
    with pytest.raises(TranslationError) as raised:
        await provider.translate(request())
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (httpx.Response(429, text="x"), QuotaExceeded, "quota"),
        (httpx.Response(400, text="bad"), ApiError, "HTTP 400"),
    ],
)
async def test_caiyun_http_errors_fail_immediately(response, error, match):
    calls = []

    def handler(http_request):
        calls.append(http_request)
        return response

    provider = build(CaiyunTranslator, handler, model=None)
    with pytest.raises(error, match=match):
        await provider.translate(request())
    assert len(calls) == 1


async def test_caiyun_segment_count_mismatch():
    provider = build(CaiyunTranslator, lambda _: reply({"target": ["only"]}), model=None)
    with pytest.raises(SegmentCountError):
        await provider.translate(request())


async def test_caiyun_missing_api_key_is_checked_before_language():
    provider = build(CaiyunTranslator, lambda _: reply({"target": ["a", "b"]}), model=None, api_key=None)
    with pytest.raises(TranslationError, match="API key is not configured"):
        await provider.translate(request(target_language=Language.BRAZILIAN_PORTUGUESE))


async def test_caiyun_unsupported_language():
    provider = build(CaiyunTranslator, lambda _: pytest.fail("no request expected"), model=None)
    with pytest.raises(UnsupportedLanguage) as raised:
        await provider.translate(request(target_language=Language.BRAZILIAN_PORTUGUESE))
    assert str(raised.value) == "caiyun does not support target language Brazilian Portuguese"


async def test_caiyun_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("CAIYUN_API_KEY", "from-env")
    seen, handler = recorder(reply({"target": ["a", "b"]}))
    await build(CaiyunTranslator, handler, model=None, api_key=None).translate(request())
    assert seen[0].headers["x-authorization"] == "token from-env"


def test_caiyun_list_models(monkeypatch):
    assert CaiyunTranslator.list_models({}) == []
    models = CaiyunTranslator.list_models({"api_key": "k"})
    assert len(models) == 1
    model = models[0]
    assert (model.provider, model.provider_name, model.model, model.name) == ("caiyun", "Caiyun", None, "Caiyun")
    assert not model.vision and not model.reasoning and model.quantizations == []
    monkeypatch.setenv("CAIYUN_API_KEY", "env")
    assert len(CaiyunTranslator.list_models({})) == 1
