"""OpenAI-compatible, OpenRouter and LM Studio translators."""

import json

import httpx
import pytest
from PIL import Image

from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.errors import ApiError, QuotaExceeded, TranslationError
from urakatsu.providers.translation.language import Language
from urakatsu.providers.translation.remote.lm_studio import LmStudioTranslator
from urakatsu.providers.translation.remote.openai_compatible import OpenAiCompatibleTranslator
from urakatsu.providers.translation.remote.openrouter import OpenRouterTranslator

ENV = {
    OpenAiCompatibleTranslator: "OPENAI_COMPATIBLE_API_KEY",
    OpenRouterTranslator: "OPENROUTER_API_KEY",
    LmStudioTranslator: "LM_STUDIO_API_KEY",
}
CHAT_URL = {
    OpenAiCompatibleTranslator: "http://localhost:11434/v1/chat/completions",
    OpenRouterTranslator: "https://openrouter.ai/api/v1/chat/completions",
    LmStudioTranslator: "http://localhost:1234/v1/chat/completions",
}
ALL = list(ENV)


def make(cls, handler, *, reasoning=True, generation=None, **settings):
    settings.setdefault("api_key", "sk-test")
    provider = cls(
        {
            "translation": {
                "model": {"provider": cls.id, "model": "vendor/model-x", "vision": True, "reasoning": reasoning},
                "generation": generation or {},
            },
            **settings,
        }
    )
    provider.transport = httpx.MockTransport(handler)
    return provider


def chat_reply(*texts):
    content = json.dumps({"translations": [{"id": i, "text": t} for i, t in enumerate(texts)]})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def request(segments=("a", "b"), **kwargs):
    return TranslationRequest(segments=list(segments), target_language=Language.ENGLISH, **kwargs)


@pytest.mark.parametrize("cls", ALL)
async def test_request_contract_and_ordered_reply(cls):
    seen = {}

    def handler(req):
        seen["req"], seen["body"] = req, json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"translations": [{"id": 1, "text": "second"}, {"id": 0, "text": "first"}]}
                            )
                        }
                    }
                ]
            },
        )

    provider = make(cls, handler, generation={"max_tokens": 512, "temperature": 0.3})
    assert await provider.translate(request()) == ["first", "second"]
    req, body = seen["req"], seen["body"]
    assert req.method == "POST"
    assert str(req.url) == CHAT_URL[cls]
    assert req.headers["authorization"] == "Bearer sk-test"
    assert list(body)[:3] == ["model", "messages", "temperature"]
    assert body["model"] == "vendor/model-x"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert isinstance(body["messages"][1]["content"], str)
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.3
    for omitted in ("top_p", "frequency_penalty", "presence_penalty", "max_completion_tokens"):
        assert omitted not in body
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "manga_translation"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["translations"]["maxItems"] == 2


@pytest.mark.parametrize("cls", [OpenAiCompatibleTranslator, LmStudioTranslator])
async def test_reasoning_effort_values(cls):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return chat_reply("x")

    await make(cls, handler, generation={"reasoning": True}).translate(request(["a"]))
    await make(cls, handler, generation={"reasoning": False}).translate(request(["a"]))
    await make(cls, handler, reasoning=False).translate(request(["a"]))
    assert bodies[0]["reasoning_effort"] == "medium"
    assert bodies[1]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in bodies[2]
    assert "reasoning" not in bodies[2]


async def test_openrouter_reasoning_and_provider_preferences():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return chat_reply("x")

    await make(OpenRouterTranslator, handler, generation={"reasoning": True}).translate(request(["a"]))
    await make(OpenRouterTranslator, handler, generation={"reasoning": False}).translate(request(["a"]))
    await make(OpenRouterTranslator, handler, reasoning=False).translate(request(["a"]))
    assert bodies[0]["reasoning"] == {"enabled": True}
    assert bodies[1]["reasoning"] == {"enabled": False}
    assert "reasoning" not in bodies[2]
    assert "reasoning_effort" not in bodies[0]
    assert bodies[0]["provider"] == {"require_parameters": True}
    keys = list(bodies[0])
    assert keys.index("reasoning") < keys.index("response_format") < keys.index("provider")


@pytest.mark.parametrize("cls", ALL)
async def test_image_is_sent_as_jpeg_data_url_after_text(cls):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return chat_reply("x")

    page = Image.new("RGB", (16, 16), "white")
    await make(cls, handler).translate(request(["a"], image=page))
    parts = seen["body"]["messages"][1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("cls", ALL)
async def test_image_dropped_when_vision_disabled(cls):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return chat_reply("x")

    provider = make(cls, handler, generation={"vision": False})
    await provider.translate(request(["a"], image=Image.new("RGB", (8, 8))))
    assert isinstance(seen["body"]["messages"][1]["content"], str)


@pytest.mark.parametrize("cls", ALL)
async def test_error_statuses(cls):
    provider = make(cls, lambda req: httpx.Response(429, text="slow down"))
    with pytest.raises(QuotaExceeded):
        await provider.translate(request())

    provider = make(cls, lambda req: httpx.Response(400, text="bad request"))
    with pytest.raises(ApiError) as caught:
        await provider.translate(request())
    assert caught.value.status == 400

    provider = make(cls, lambda req: httpx.Response(402, text='{"error":"insufficient_quota"}'))
    with pytest.raises(QuotaExceeded):
        await provider.translate(request())


@pytest.mark.parametrize("cls", ALL)
async def test_missing_choices_and_content(cls):
    provider = make(cls, lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(TranslationError, match="returned no choices"):
        await provider.translate(request())

    provider = make(cls, lambda req: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    with pytest.raises(TranslationError, match="returned no message content"):
        await provider.translate(request())

    provider = make(cls, lambda req: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(TranslationError, match="missing field `choices`"):
        await provider.translate(request())


@pytest.mark.parametrize("cls", ALL)
async def test_invalid_translation_json_fails_without_retry(cls):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    with pytest.raises(TranslationError, match="invalid translation JSON"):
        await make(cls, handler).translate(request())
    assert len(calls) == 1


@pytest.mark.parametrize("cls", ALL)
async def test_missing_ids_keep_source_text(cls):
    provider = make(cls, lambda req: chat_reply("only first"))
    assert await provider.translate(request(["a", "b"])) == ["only first", "b"]


async def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = make(OpenRouterTranslator, lambda req: chat_reply("x"), api_key=None)
    with pytest.raises(TranslationError, match="openrouter API key is not configured"):
        await provider.translate(request(["a"]))


async def test_openrouter_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        return chat_reply("x")

    await make(OpenRouterTranslator, handler, api_key=None).translate(request(["a"]))
    assert seen["auth"] == "Bearer from-env"


@pytest.mark.parametrize("cls", [OpenAiCompatibleTranslator, LmStudioTranslator])
async def test_key_is_optional_and_authorization_omitted(cls, monkeypatch):
    monkeypatch.delenv(ENV[cls], raising=False)
    seen = {}

    def handler(req):
        seen["headers"] = req.headers
        return chat_reply("x")

    assert await make(cls, handler, api_key=None).translate(request(["a"])) == ["x"]
    assert "authorization" not in seen["headers"]


@pytest.mark.parametrize(
    ("cls", "base_url", "expected"),
    [
        (OpenAiCompatibleTranslator, "http://example.test:8080/v1/", "http://example.test:8080/v1/chat/completions"),
        (LmStudioTranslator, "http://example.test:9000/", "http://example.test:9000/v1/chat/completions"),
    ],
)
async def test_base_url_override(cls, base_url, expected):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return chat_reply("x")

    await make(cls, handler, base_url=base_url).translate(request(["a"]))
    assert seen["url"] == expected


async def test_model_selection_is_required():
    provider = OpenRouterTranslator(
        {"api_key": "k", "translation": {"model": {"provider": "openrouter", "model": None}}}
    )
    provider.transport = httpx.MockTransport(lambda req: chat_reply("x"))
    with pytest.raises(TranslationError, match="requires a selected model"):
        await provider.translate(request(["a"]))


def test_openai_compatible_list_models(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    seen = {}

    def handler(req):
        seen["req"] = req
        return httpx.Response(200, json={"data": [{"id": "llama-3.1_8b"}, {"id": "org/qwen-vl"}]})

    models = OpenAiCompatibleTranslator.list_models(
        {"base_url": "http://example.test/v1/"}, transport=httpx.MockTransport(handler)
    )
    assert str(seen["req"].url) == "http://example.test/v1/models"
    assert "authorization" not in seen["req"].headers
    assert [(m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("llama-3.1_8b", "Llama 3.1 8b", True, True),
        ("org/qwen-vl", "Qwen Vl", True, True),
    ]
    assert all(m.provider == "openai-compatible" and m.provider_name == "OpenAI-compatible" for m in models)


def test_openai_compatible_list_models_sends_key_and_propagates_errors():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(500, text="down")

    with pytest.raises(ApiError) as caught:
        OpenAiCompatibleTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(handler))
    assert caught.value.status == 500
    assert seen["auth"] == "Bearer k"


def test_openrouter_list_models(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert OpenRouterTranslator.list_models({}, transport=httpx.MockTransport(lambda r: pytest.fail("called"))) == []

    seen = {}
    data = {
        "data": [
            {
                "id": "p/reasoner",
                "name": "Reasoner",
                "architecture": {"input_modalities": ["text"]},
                "supported_parameters": [],
                "reasoning": {"supported_efforts": ["high"]},
            },
            {
                "id": "p/vision",
                "name": "Vision (free)",
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["reasoning"],
            },
            {
                "id": "p/plain",
                "name": "Plain",
                "architecture": {"input_modalities": ["text"]},
                "supported_parameters": ["tools"],
            },
        ]
    }

    def handler(req):
        seen["req"] = req
        return httpx.Response(200, json=data)

    models = OpenRouterTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(handler))
    assert str(seen["req"].url) == "https://openrouter.ai/api/v1/models"
    assert seen["req"].headers["authorization"] == "Bearer k"
    assert [(m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("p/reasoner", "Reasoner", False, True),
        ("p/vision", "Vision (free)", True, True),
        ("p/plain", "Plain", False, False),
    ]


def test_openrouter_list_models_swallows_failures():
    for response in (
        httpx.Response(500, text="down"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"data": [{"id": "only-id"}]}),
    ):
        models = OpenRouterTranslator.list_models(
            {"api_key": "k"}, transport=httpx.MockTransport(lambda req, r=response: r)
        )
        assert models == []


def test_lm_studio_list_models_filters_kinds(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    seen = {}
    data = {
        "models": [
            {"type": "llm", "key": "publisher/chat-model", "capabilities": {"vision": False}},
            {
                "type": "llm",
                "key": "publisher/vision-model",
                "capabilities": {"vision": True, "reasoning": {"allowed_options": ["off", "on"], "default": "on"}},
            },
            {"type": "vlm", "key": "publisher/no-caps"},
            {"type": "embedding", "key": "publisher/embed-model"},
        ]
    }

    def handler(req):
        seen["req"] = req
        return httpx.Response(200, json=data)

    models = LmStudioTranslator.list_models({}, transport=httpx.MockTransport(handler))
    assert str(seen["req"].url) == "http://localhost:1234/api/v1/models"
    assert "authorization" not in seen["req"].headers
    assert [(m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("publisher/chat-model", "Chat Model", False, False),
        ("publisher/vision-model", "Vision Model", True, True),
        ("publisher/no-caps", "No Caps", False, False),
    ]


def test_lm_studio_list_models_uses_base_url_key_and_propagates_errors():
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers.get("authorization")
        return httpx.Response(401, text="no")

    with pytest.raises(ApiError) as caught:
        LmStudioTranslator.list_models(
            {"base_url": "http://example.test:5000/", "api_key": "k"}, transport=httpx.MockTransport(handler)
        )
    assert caught.value.status == 401
    assert seen == {"url": "http://example.test:5000/api/v1/models", "auth": "Bearer k"}
