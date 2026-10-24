"""DeepSeek, Grok and MiniMax translation providers."""

import json

import httpx
import pytest
from PIL import Image

from urakatsu.providers.translation import prompt
from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.errors import (
    ApiError,
    QuotaExceeded,
    SegmentCountError,
    TranslationError,
)
from urakatsu.providers.translation.language import Language
from urakatsu.providers.translation.remote.deepseek import DeepSeekTranslator
from urakatsu.providers.translation.remote.grok import GrokTranslator
from urakatsu.providers.translation.remote.minimax import MiniMaxTranslator

ENV = {"deepseek": "DEEPSEEK_API_KEY", "grok": "XAI_API_KEY", "minimax": "MINIMAX_API_KEY"}
CLASSES = {"deepseek": DeepSeekTranslator, "grok": GrokTranslator, "minimax": MiniMaxTranslator}
MODELS = {"deepseek": "deepseek-v4-pro", "grok": "grok-4.5", "minimax": "MiniMax-M3"}
ALL = list(CLASSES)


def reply_json(*texts):
    return json.dumps({"translations": [{"id": i, "text": t} for i, t in enumerate(texts)]})


def success_body(provider, text):
    """A well-formed provider reply carrying ``text`` as the model output."""
    if provider == "grok":
        return {"output": [{"content": [{"type": "output_text", "text": text}]}]}
    body = {"choices": [{"message": {"content": text}}]}
    if provider == "minimax":
        body["base_resp"] = {"status_code": 0, "status_msg": "success"}
    return body


def make(provider, *, model=None, generation=None, api_key="secret", vision=True, reasoning=True):
    settings = {
        "translation": {
            "model": {
                "provider": provider,
                "model": model or MODELS[provider],
                "vision": vision,
                "reasoning": reasoning,
            },
            "generation": generation or {},
        }
    }
    if api_key is not None:
        settings["api_key"] = api_key
    return CLASSES[provider](settings)


def serve(provider_obj, status=200, body=None, text=None):
    """Mock the endpoint; returns the list of captured requests."""
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=body)

    provider_obj.transport = httpx.MockTransport(handler)
    return captured


def request(*segments, **kwargs):
    return TranslationRequest(list(segments), Language.JAPANESE, **kwargs)


def sent(captured):
    return json.loads(captured[0].content)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV.values():
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- DeepSeek


async def test_deepseek_request_contract():
    provider = make("deepseek", generation={"max_tokens": 1024, "top_p": 0.9, "reasoning": False})
    captured = serve(provider, body=success_body("deepseek", reply_json("a", "b")))
    result = await provider.translate(request("one", "two"))
    assert result == ["a", "b"]
    http = captured[0]
    assert http.method == "POST"
    assert str(http.url) == "https://api.deepseek.com/chat/completions"
    assert http.headers["authorization"] == "Bearer secret"
    body = sent(captured)
    assert list(body) == ["model", "messages", "thinking", "max_tokens", "response_format", "temperature", "top_p"]
    assert body["model"] == "deepseek-v4-pro"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert isinstance(body["messages"][1]["content"], str)
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 1024
    assert body["top_p"] == 0.9
    assert body["temperature"] == 1.3
    assert body["response_format"] == {"type": "json_object"}
    assert "frequency_penalty" not in body and "presence_penalty" not in body and "json_schema" not in body


async def test_deepseek_omits_unset_options_and_honours_temperature():
    provider = make("deepseek", generation={"temperature": 0.4, "reasoning": True})
    captured = serve(provider, body=success_body("deepseek", reply_json("a")))
    await provider.translate(request("one"))
    body = sent(captured)
    assert body["thinking"] == {"type": "enabled"}
    assert body["temperature"] == 0.4
    assert "max_tokens" not in body and "top_p" not in body

    # A model without reasoning support drops the toggle entirely.
    provider = make("deepseek", reasoning=False)
    captured = serve(provider, body=success_body("deepseek", reply_json("a")))
    await provider.translate(request("one"))
    assert "thinking" not in sent(captured)


async def test_deepseek_sends_image_as_data_url_part():
    provider = make("deepseek")
    captured = serve(provider, body=success_body("deepseek", reply_json("a")))
    await provider.translate(request("one", image=Image.new("RGB", (8, 8), "white")))
    content = sent(captured)["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_deepseek_reply_shape_errors():
    provider = make("deepseek")
    serve(provider, body={"choices": []})
    with pytest.raises(TranslationError, match="DeepSeek returned no choices"):
        await provider.translate(request("one"))
    serve(provider, body={"choices": [{"message": {"content": None}}]})
    with pytest.raises(TranslationError, match="DeepSeek returned no message content"):
        await provider.translate(request("one"))
    serve(provider, body={"nope": 1})
    with pytest.raises(TranslationError, match="missing field `choices`"):
        await provider.translate(request("one"))


def test_deepseek_list_models():
    captured = []

    def handler(http_request):
        captured.append(http_request)
        return httpx.Response(200, json={"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})

    models = DeepSeekTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(handler))
    assert str(captured[0].url) == "https://api.deepseek.com/models"
    assert captured[0].method == "GET"
    assert captured[0].headers["authorization"] == "Bearer k"
    assert [(m.provider, m.provider_name, m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("deepseek", "DeepSeek", "deepseek-chat", "Deepseek Chat", False, True),
        ("deepseek", "DeepSeek", "deepseek-reasoner", "Deepseek Reasoner", False, True),
    ]


# -------------------------------------------------------------------- Grok


async def test_grok_request_contract_with_image():
    provider = make("grok", generation={"max_tokens": 1024, "reasoning": True, "temperature": 0.2})
    captured = serve(provider, body=success_body("grok", reply_json("a", "b")))
    result = await provider.translate(request("one", "two", image=Image.new("RGB", (8, 8), "white")))
    assert result == ["a", "b"]
    http = captured[0]
    assert http.method == "POST"
    assert str(http.url) == "https://api.x.ai/v1/responses"
    assert http.headers["authorization"] == "Bearer secret"
    body = sent(captured)
    assert list(body) == ["model", "input", "store", "temperature", "max_output_tokens", "reasoning", "text"]
    assert body["model"] == "grok-4.5"
    assert body["store"] is False
    assert body["max_output_tokens"] == 1024
    assert body["reasoning"] == {"effort": "high"}
    fmt = body["text"]["format"]
    assert (fmt["type"], fmt["name"], fmt["strict"]) == ("json_schema", "manga_translation", True)
    assert fmt["schema"] == prompt.output_schema(2)
    assert [m["role"] for m in body["input"]] == ["system", "user"]
    content = body["input"][1]["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "top_p" not in body


async def test_grok_omits_reasoning_when_off_and_sends_plain_text_without_image():
    provider = make("grok", generation={"reasoning": False})
    captured = serve(provider, body=success_body("grok", reply_json("a")))
    await provider.translate(request("one"))
    body = sent(captured)
    assert "reasoning" not in body
    assert "temperature" not in body and "max_output_tokens" not in body
    assert isinstance(body["input"][1]["content"], str)


async def test_grok_takes_first_output_text_and_reports_missing_text():
    provider = make("grok")
    serve(
        provider,
        body={
            "output": [
                {"type": "reasoning"},
                {"content": [{"type": "refusal", "text": "no"}, {"type": "output_text", "text": reply_json("ok")}]},
                {"content": [{"type": "output_text", "text": reply_json("later")}]},
            ]
        },
    )
    assert await provider.translate(request("one")) == ["ok"]
    serve(provider, body={"output": [{"content": [{"type": "output_text"}]}]})
    with pytest.raises(TranslationError, match="Grok returned no output text"):
        await provider.translate(request("one"))
    serve(provider, body={"choices": []})
    with pytest.raises(TranslationError, match="missing field `output`"):
        await provider.translate(request("one"))


def test_grok_list_models_reports_vision_from_input_modalities():
    captured = []

    def handler(http_request):
        captured.append(http_request)
        return httpx.Response(
            200,
            json={
                "models": [
                    {"id": "grok-4.5", "input_modalities": ["text", "image"]},
                    {"id": "grok-code-fast", "input_modalities": ["text"]},
                ]
            },
        )

    models = GrokTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(handler))
    assert str(captured[0].url) == "https://api.x.ai/v1/language-models"
    assert captured[0].headers["authorization"] == "Bearer k"
    assert [(m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("grok-4.5", "Grok 4.5", True, True),
        ("grok-code-fast", "Grok Code Fast", False, True),
    ]


# ----------------------------------------------------------------- MiniMax


async def test_minimax_native_request_without_schema_for_other_models():
    provider = make("minimax", generation={"temperature": 0.7, "top_p": 0.95, "max_tokens": 1024})
    captured = serve(provider, body=success_body("minimax", reply_json("a", "b")))
    assert await provider.translate(request("one", "two")) == ["a", "b"]
    http = captured[0]
    assert http.method == "POST"
    assert str(http.url) == "https://api.minimax.io/v1/text/chatcompletion_v2"
    assert http.headers["authorization"] == "Bearer secret"
    body = sent(captured)
    assert list(body) == ["model", "messages", "temperature", "top_p", "max_completion_tokens"]
    assert body["model"] == "MiniMax-M3"
    assert body["max_completion_tokens"] == 1024
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert all(isinstance(m["content"], str) for m in body["messages"])
    assert "response_format" not in body


async def test_minimax_text_01_sends_strict_json_schema():
    provider = make("minimax", model="MiniMax-Text-01")
    captured = serve(provider, body=success_body("minimax", reply_json("a", "b")))
    await provider.translate(request("one", "two"))
    body = sent(captured)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "manga_translation", "strict": True, "schema": prompt.output_schema(2)},
    }
    assert body["response_format"]["json_schema"]["schema"]["properties"]["translations"]["maxItems"] == 2
    assert "temperature" not in body and "top_p" not in body and "max_completion_tokens" not in body


async def test_minimax_base_resp_failure_and_shape_errors():
    provider = make("minimax")
    serve(provider, body={"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})
    with pytest.raises(TranslationError, match="MiniMax returned status 1008: insufficient balance"):
        await provider.translate(request("one"))
    serve(provider, body={"choices": []})
    with pytest.raises(TranslationError, match="MiniMax returned no choices"):
        await provider.translate(request("one"))
    # An absent base_resp counts as status 0.
    serve(provider, body={"choices": [{"message": {"content": reply_json("ok")}}]})
    assert await provider.translate(request("one")) == ["ok"]


def test_minimax_list_models():
    captured = []

    def handler(http_request):
        captured.append(http_request)
        return httpx.Response(200, json={"data": [{"id": "MiniMax-M3"}, {"id": "MiniMax-Text-01"}]})

    models = MiniMaxTranslator.list_models({"api_key": "k"}, transport=httpx.MockTransport(handler))
    assert str(captured[0].url) == "https://api.minimax.io/v1/models"
    assert captured[0].headers["authorization"] == "Bearer k"
    assert [(m.model, m.name, m.vision, m.reasoning) for m in models] == [
        ("MiniMax-M3", "MiniMax M3", False, True),
        ("MiniMax-Text-01", "MiniMax Text 01", False, True),
    ]


# ------------------------------------------------------------ shared policy


@pytest.mark.parametrize("provider", ALL)
async def test_reply_is_restored_to_segment_order(provider):
    obj = make(provider)
    serve(obj, body=success_body(provider, json.dumps({"translations": [{"id": 1, "text": "two"}, {"id": 0, "text": "one"}]})))
    assert await obj.translate(request("a", "b")) == ["one", "two"]


@pytest.mark.parametrize("provider", ALL)
async def test_quota_and_api_errors(provider):
    obj = make(provider)
    serve(obj, status=429, text="slow down")
    with pytest.raises(QuotaExceeded):
        await obj.translate(request("one"))
    serve(obj, status=403, text="Your credit balance is too low")
    with pytest.raises(QuotaExceeded):
        await obj.translate(request("one"))
    serve(obj, status=400, text="bad request")
    with pytest.raises(ApiError) as caught:
        await obj.translate(request("one"))
    assert caught.value.status == 400
    assert "HTTP 400" in str(caught.value)


@pytest.mark.parametrize("provider", ALL)
async def test_invalid_json_body_is_a_decode_error(provider):
    obj = make(provider)
    serve(obj, text="not json")
    with pytest.raises(TranslationError, match=f"failed to decode {provider} response"):
        await obj.translate(request("one"))


@pytest.mark.parametrize("provider", ALL)
async def test_no_retry_on_failure(provider):
    obj = make(provider)
    captured = serve(obj, status=500, text="boom")
    with pytest.raises(ApiError):
        await obj.translate(request("one"))
    assert len(captured) == 1


@pytest.mark.parametrize("provider", ALL)
async def test_segment_count_mismatch_is_rejected(provider, monkeypatch):
    # prompt.translations always restores one entry per segment, so the guard in
    # translate() is only reachable through a parser that returns the wrong length.
    obj = make(provider)
    serve(obj, body=success_body(provider, reply_json("a")))
    monkeypatch.setattr(prompt, "translations", lambda *args: ["only one"])
    with pytest.raises(SegmentCountError):
        await obj.translate(request("one", "two"))


@pytest.mark.parametrize("provider", ALL)
async def test_malformed_translation_json_fails_closed(provider):
    obj = make(provider)
    serve(obj, body=success_body(provider, "this is not json"))
    with pytest.raises(TranslationError, match="invalid translation JSON"):
        await obj.translate(request("one"))


@pytest.mark.parametrize("provider", ALL)
async def test_missing_api_key(provider):
    obj = make(provider, api_key=None)
    captured = serve(obj, body=success_body(provider, reply_json("a")))
    with pytest.raises(TranslationError, match=f"{provider} API key is not configured"):
        await obj.translate(request("one"))
    assert captured == []


@pytest.mark.parametrize("provider", ALL)
async def test_api_key_from_environment(provider, monkeypatch):
    monkeypatch.setenv(ENV[provider], "from-env")
    obj = make(provider, api_key=None)
    captured = serve(obj, body=success_body(provider, reply_json("a")))
    await obj.translate(request("one"))
    assert captured[0].headers["authorization"] == "Bearer from-env"


@pytest.mark.parametrize("provider", ALL)
def test_registration_and_env_key(provider):
    cls = CLASSES[provider]
    assert cls.id == provider
    assert cls.env_key == ENV[provider]


@pytest.mark.parametrize("provider", ALL)
def test_list_models_empty_without_key_and_propagates_errors(provider, monkeypatch):
    cls = CLASSES[provider]

    def explode(http_request):
        raise AssertionError("no request expected without a key")

    assert cls.list_models({}, transport=httpx.MockTransport(explode)) == []
    monkeypatch.setenv(ENV[provider], "env-key")
    with pytest.raises(QuotaExceeded):
        cls.list_models({}, transport=httpx.MockTransport(lambda r: httpx.Response(429, text="x")))
    with pytest.raises(ApiError):
        cls.list_models({"api_key": "k"}, transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
