"""Agents: the tool loop, Urakatsu's tools, the wire formats, ChatGPT sign-in, and the routes."""

import asyncio
import base64
import io
import json
import time

import httpx
import pytest
from PIL import Image

from urakatsu.agents import AgentService, get_agent
from urakatsu.agents.backends.anthropic import AnthropicBackend
from urakatsu.agents.backends.base import AgentError, ChatBackend, failure_kind
from urakatsu.agents.backends.openai_chat import OpenAiChatBackend
from urakatsu.agents.chatgpt import CLIENT_ID, ChatGptAuth, ChatGptBackend, authorize_url, jwt_claims, pkce_pair
from urakatsu.agents.image_edit import Box, ChatGptImageEditor, ImageEditor, edit_region, pad_to, plan_window
from urakatsu.agents.commands import parse as parse_command
from urakatsu.agents.compaction import DEFAULT_CONTEXT_WINDOW, OMITTED, ContextBudget, estimate_tokens, fit
from urakatsu.agents.messages import Message, TextDelta, ToolCall, ToolCallRequest, ToolSpec
from urakatsu.agents.models import ModelDirectory
from urakatsu.agents.session import ConversationStore, Turns
from urakatsu.agents.tools import ToolContext
from urakatsu.core.registry import registry
from urakatsu.pipeline import Pipeline
from urakatsu.providers.translation.errors import ApiError, QuotaExceeded
from urakatsu.runtimes import RuntimeManager
from urakatsu.service.app import build_app
from urakatsu.service.servicer import UrakatsuServicer
from urakatsu.service.workspace import Workspace

from fakes import FakeDetector, FakeInpainter, FakeOcr, FakeTranslator, fake_config, fake_processors, page_png


@pytest.fixture
def workspace(isolated_home):
    fakes = (FakeDetector, FakeOcr, FakeTranslator, FakeInpainter)
    registry.discover()
    for cls in fakes:
        registry.add(cls)
    ws = Workspace(device="cpu")
    config = fake_config()
    ws.pipeline = Pipeline(config, RuntimeManager("cpu"), registry, fake_processors())
    ws.set_config(config)
    yield ws
    for cls in fakes:
        registry.remove(cls.stage, cls.id)


@pytest.fixture
def project(workspace):
    project = workspace.create_project("chat").id
    page = workspace.add_page(project, "p.png", page_png()).id
    for bounds in ({"x": 10, "y": 10, "width": 40, "height": 20}, {"x": 60, "y": 60, "width": 40, "height": 20}):
        workspace.create_text_block(project, page, bounds)
    return project, page


class ScriptedBackend(ChatBackend):
    """Plays back one list of events per model step and records what it was sent."""

    def __init__(self, steps, ref="test:model"):
        self.steps = list(steps)
        self.ref = ref
        self.seen = []

    async def stream(self, system, messages, tools):
        self.seen.append((system, [m.to_json() for m in messages], [t.name for t in tools]))
        step = self.steps.pop(0)
        if callable(step):
            step = await step()
        for event in step:
            yield event


def call(name, arguments, id="c1"):
    return ToolCallRequest(id, name, json.dumps(arguments))


async def run(agents, text, backend, project=None, page=None, conversation=None):
    async def resolve(ref):
        return backend

    agents.models.backend = resolve
    conversation_id = await agents.send(conversation, text, backend.ref, project_id=project, page_id=page)
    events = [event async for event in agents.turns.events(conversation_id)]
    return conversation_id, events


# -- loop and tools -------------------------------------------------------------------------------


async def test_turn_reads_then_edits_then_answers(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    first = workspace.page(project_id, page_id).blocks[0].layer_id
    backend = ScriptedBackend(
        [
            [TextDelta("Looking. "), call("get_page", {})],
            [call("edit_text_blocks", {"edits": [{"layer_id": first, "translation": "Hello!"}]}, "c2")],
            [TextDelta("Done.")],
        ]
    )
    conversation_id, events = await run(agents, "Translate the first bubble", backend, project_id, page_id)

    assert [e["type"] for e in events if e["type"] in ("tool_call", "tool_result", "done")] == [
        "tool_call", "tool_result", "tool_call", "tool_result", "done"
    ]  # fmt: skip
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["ok"] and json.loads(results[0]["result"])["blocks"][0]["layer_id"] == first
    assert results[1]["changed"] == ["project"]
    assert events[-1] == {"type": "done", "status": "done"}

    block = workspace.page(project_id, page_id).blocks[0]
    assert block.translation == "Hello!" and block.translation_origin == "user"

    saved = agents.conversation(conversation_id)
    assert saved["title"] == "Translate the first bubble" and not saved["running"]
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert saved["messages"][-1]["content"] == "Done." and saved["messages"][-1]["model"] == "test:model"
    # The instructions name the open project and page.
    assert page_id in backend.seen[0][0] and "edit_text_blocks" in backend.seen[0][2]


async def test_tool_errors_go_back_to_the_model(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    backend = ScriptedBackend([[call("get_page", {"page_id": "nope"}), call("no_such_tool", {}, "c2")], [TextDelta("Sorry.")]])
    conversation_id, events = await run(agents, "hi", backend, project_id, page_id)
    results = [e for e in events if e["type"] == "tool_result"]
    assert [r["ok"] for r in results] == [False, False]
    assert "unknown tool" in json.loads(results[1]["result"])["error"]
    assert events[-1]["status"] == "done"


async def test_malformed_arguments_are_reported(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    backend = ScriptedBackend([[ToolCallRequest("c1", "get_page", "{oops")], [TextDelta("ok")]])
    _, events = await run(agents, "hi", backend, project_id, page_id)
    [result] = [e for e in events if e["type"] == "tool_result"]
    assert not result["ok"] and "not valid JSON" in result["result"]


async def test_run_pipeline_tool_waits_and_streams_progress(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    backend = ScriptedBackend([[call("run_pipeline", {"mode": "full", "scope": "page"})], [TextDelta("Ran it.")]])
    _, events = await run(agents, "run it", backend, project_id, page_id)
    [result] = [e for e in events if e["type"] == "tool_result"]
    assert json.loads(result["result"])["status"] == "completed"
    assert any(e["type"] == "progress" for e in events)


async def test_stop_keeps_the_transcript_valid(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    gate = asyncio.Event()

    async def hang():
        await gate.wait()
        return []

    backend = ScriptedBackend([[call("get_page", {})], hang])

    async def resolve(ref):
        return backend

    agents.models.backend = resolve
    conversation_id = await agents.send(None, "hi", backend.ref, project_id=project_id, page_id=page_id)
    for _ in range(100):
        if len(backend.seen) == 2:
            break
        await asyncio.sleep(0.01)
    agents.turns.stop(conversation_id)
    events = [event async for event in agents.turns.events(conversation_id)]
    assert events[-1] == {"type": "done", "status": "stopped"}
    messages = agents.conversation(conversation_id)["messages"]
    calls = {c["id"] for m in messages for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert calls == answered


async def test_follow_up_continues_the_conversation(workspace, project):
    project_id, page_id = project
    agents = AgentService(workspace)
    conversation_id, _ = await run(agents, "one", ScriptedBackend([[TextDelta("a")]]), project_id, page_id)
    second = ScriptedBackend([[TextDelta("b")]], ref="other:model")
    await run(agents, "two", second, project_id, page_id, conversation_id)
    assert [m["role"] for m in second.seen[0][1]] == ["user", "assistant", "user"]
    assert agents.conversation(conversation_id)["model"] == "other:model"
    assert [c["id"] for c in agents.store.list(project_id)] == [conversation_id]


def test_app_tools_edit_like_the_ui(workspace, project):
    project_id, page_id = project
    tools = get_agent("assistant").toolset()
    context = ToolContext(workspace, project_id, page_id)
    first, second = [b.layer_id for b in workspace.page(project_id, page_id).blocks]

    async def go(name, args):
        ok, text = await tools.call(context, name, args)
        assert ok, text
        return json.loads(text)

    async def scenario():
        await go("set_typography", {"layer_ids": [first], "font": "Test Sans", "size": 14, "color": "#ff0000"})
        typography = workspace.page(project_id, page_id).blocks[0].typography
        assert (typography.preferred_font, typography.size, typography.auto_fit, typography.color) == ("Test Sans", 14, False, "#ff0000ff")

        await go("edit_text_blocks", {"edits": [{"layer_id": second, "translation": "Kept", "source_text": "元"}]})
        hits = await go("search_text", {"query": "kept"})
        assert [h["layer_id"] for h in hits["hits"]] == [second]

        created = await go("create_text_block", {"x": 5, "y": 5, "width": 30, "height": 30, "translation": "New"})
        assert workspace.page(project_id, page_id).blocks[-1].translation == "New"
        await go("delete_text_blocks", {"layer_ids": [created["layer_id"]]})
        await go("undo", {})
        assert len(workspace.page(project_id, page_id).blocks) == 3

        overview = await go("get_project", {})
        assert overview["chapters"][0]["pages"][0]["open"] is True

        settings = await go("configure_translation", {"target_language": "fr-FR", "instructions": "Keep honorifics."})
        assert settings["target_language"] == "fr-FR" and workspace.config.translation.instructions == "Keep honorifics."
        assert "config" in context.changed

        events = []
        context.emit = events.append
        await go("open_page", {"page_id": page_id})
        assert events == [{"type": "ui", "action": "open_page", "page_id": page_id}]

    asyncio.run(scenario())


# -- context budget and retries -------------------------------------------------------------------


def turn(n, result_size, calls=1):
    """One turn: the user asks, the model calls ``get_page`` ``calls`` times, then answers."""
    messages = [Message("user", f"question {n}")]
    for i in range(calls):
        messages.append(Message("assistant", "", [ToolCall(f"c{n}{i}", "get_page", {})]))
        messages.append(Message("tool", "x" * result_size, tool_call_id=f"c{n}{i}", name="get_page"))
    messages.append(Message("assistant", f"answer {n}"))
    return messages


def test_fit_leaves_a_conversation_that_fits_alone():
    messages = turn(1, 300)
    assert fit(messages, 10_000) is messages


def test_fit_shortens_older_results_first_and_spares_the_newest_step():
    messages = turn(1, 30_000) + [Message("user", "again"), Message("assistant", "", [ToolCall("n", "get_page", {})])]
    messages.append(Message("tool", "y" * 3_000, tool_call_id="n", name="get_page"))
    fitted = fit(messages, 3_000)
    old, newest = fitted[2].content, fitted[-1].content
    assert len(old) < 1_000 and "call get_page again" in old and old.startswith("xxx")
    assert newest == "y" * 3_000
    assert messages[2].content == "x" * 30_000  # the saved conversation is untouched


def test_fit_splits_what_is_left_between_the_newest_results():
    messages = [Message("user", "go"), Message("assistant", "", [ToolCall("a", "get_page", {}), ToolCall("b", "search_text", {})])]
    messages += [Message("tool", "a" * 30_000, tool_call_id="a", name="get_page"), Message("tool", "b" * 30_000, tool_call_id="b", name="search_text")]
    fitted = fit(messages, 4_000)
    assert sum(estimate_tokens(m.content) for m in fitted) <= 4_000
    assert all(1_000 < estimate_tokens(m.content) <= 2_000 for m in fitted[2:])
    assert "call search_text again" in fitted[3].content


def test_fit_leaves_out_whole_old_turns_but_keeps_the_last():
    messages = turn(1, 20_000) + turn(2, 20_000) + [Message("user", "z" * 9_000)]
    fitted = fit(messages, 3_000)
    assert fitted[0].role == "user" and fitted[0].content.startswith(OMITTED) and fitted[-1].content.endswith("z" * 9_000)
    assert not any(m.content.startswith("question 1") for m in fitted)
    # Every tool call that is kept still has its result.
    calls = {c.id for m in fitted for c in m.tool_calls}
    assert calls == {m.tool_call_id for m in fitted if m.role == "tool"}


def test_budget_reserves_the_reply_and_halves_on_overflow():
    budget = ContextBudget(8192)
    assert budget.for_messages("", []) == 8192 - 2048
    budget.shrink()
    assert budget.window == 4096
    assert ContextBudget(None).window == DEFAULT_CONTEXT_WINDOW


def test_failure_kinds():
    assert failure_kind(ApiError("p", 503, "busy")) == "transient"
    assert failure_kind(ApiError("p", 500, "oops")) == "transient"
    assert failure_kind(ApiError("p", 401, "bad key")) is None
    assert failure_kind(ApiError("p", 400, "bad request")) is None
    assert failure_kind(ApiError("p", 400, '{"code": "context_length_exceeded"}')) == "overflow"
    assert failure_kind(ApiError("p", 400, "prompt is too long: 300000 tokens > 200000 maximum")) == "overflow"
    assert failure_kind(QuotaExceeded("p", 429, "rate limit exceeded, slow down")) == "transient"
    assert failure_kind(QuotaExceeded("p", 429, '{"code": "insufficient_quota"}')) is None
    assert failure_kind(QuotaExceeded("p")) is None
    assert failure_kind(AgentError("x", transient=True)) == "transient"
    assert failure_kind(ValueError("x")) is None


def failing(error, *before):
    """A model step that streams ``before`` and then fails with ``error``."""

    def events():
        yield from before
        raise error

    return events()


@pytest.fixture
def no_waits(monkeypatch):
    from urakatsu.agents import session

    monkeypatch.setattr(session, "RETRY_DELAYS", (0.0, 0.0, 0.0))


async def test_transient_failure_is_sent_again(workspace, project, no_waits):
    project_id, page_id = project
    agents = AgentService(workspace)
    backend = ScriptedBackend([failing(ApiError("test", 503, "busy"), TextDelta("Hal")), [TextDelta("Hello.")]])
    conversation_id, events = await run(agents, "hi", backend, project_id, page_id)
    assert [e["type"] for e in events if e["type"] in ("text", "retry", "done")] == ["text", "retry", "text", "done"]
    [retry] = [e for e in events if e["type"] == "retry"]
    assert retry["reason"] == "transient" and "HTTP 503" in retry["message"]
    assert events[-1]["status"] == "done"
    assert [m["content"] for m in agents.conversation(conversation_id)["messages"]] == ["hi", "Hello."]


async def test_retries_run_out_and_spent_quotas_are_not_retried(workspace, project, no_waits):
    project_id, page_id = project
    agents = AgentService(workspace)
    busy = ScriptedBackend([failing(ApiError("test", 502, "bad gateway")) for _ in range(4)])
    _, events = await run(agents, "hi", busy, project_id, page_id)
    assert len(busy.seen) == 4 and events[-1]["status"] == "failed"

    spent = ScriptedBackend([failing(QuotaExceeded("test", 429, "insufficient_quota"))])
    _, events = await run(agents, "hi", spent, project_id, page_id)
    assert len(spent.seen) == 1 and events[-1]["status"] == "failed"
    assert not any(e["type"] == "retry" for e in events)


async def test_steps_are_fitted_to_the_window_and_shrink_on_overflow(workspace, project, no_waits):
    project_id, page_id = project
    agents = AgentService(workspace)
    conversation = agents.store.create("assistant", "test:model", project_id)
    conversation.messages = turn(1, 120_000)
    agents.store.save(conversation)

    small = ScriptedBackend([[TextDelta("ok")]])
    small.context_window = 8192
    await run(agents, "next", small, project_id, page_id, conversation.id)
    sent = small.seen[0][1]
    assert len(sent[2]["content"]) < 1_000 and "shortened" in sent[2]["content"]
    assert len(agents.conversation(conversation.id)["messages"][2]["content"]) == 120_000

    overflow = ApiError("test", 400, "This model's maximum context length is 20000 tokens")
    large = ScriptedBackend([failing(overflow), failing(overflow), [TextDelta("ok")]])
    _, events = await run(agents, "again", large, project_id, page_id, conversation.id)
    assert [e["reason"] for e in events if e["type"] == "retry"] == ["overflow", "overflow"]
    assert len(large.seen[0][1][2]["content"]) == 120_000  # 40k tokens fit the assumed 128k window
    assert len(large.seen[2][1][2]["content"]) < 1_000  # but not a quarter of it
    assert events[-1]["status"] == "done"


async def test_connection_errors_and_overload_events_are_transient():
    def refuse(request):
        raise httpx.ConnectError("refused")

    backend = OpenAiChatBackend("openai", "https://api.test/chat", "m", "key")
    backend.transport = httpx.MockTransport(refuse)
    with pytest.raises(AgentError) as caught:
        await collect(backend)
    assert failure_kind(caught.value) == "transient"

    claude = AnthropicBackend("claude-x", "key")
    claude.transport = httpx.MockTransport(lambda r: sse(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})))
    with pytest.raises(AgentError) as caught:
        await collect(claude)
    assert failure_kind(caught.value) == "transient"


def test_chatgpt_backend_budgets_the_listed_window(tmp_path):
    backend = ChatGptBackend(ChatGptAuth(tmp_path / "auth.json"), "m", {"context_window": 272_000, "effective_context_window_percent": 95})
    assert backend.context_window == 258_400
    assert ChatGptBackend(ChatGptAuth(tmp_path / "auth.json"), "m").context_window is None


# -- wire formats ---------------------------------------------------------------------------------


def sse(*events):
    lines = []
    for event in events:
        if isinstance(event, tuple):
            lines.append(f"event: {event[0]}\ndata: {json.dumps(event[1])}\n")
        else:
            lines.append(f"data: {event if isinstance(event, str) else json.dumps(event)}\n")
    return httpx.Response(200, text="\n".join(lines) + "\n", headers={"content-type": "text/event-stream"})


TOOLS = [ToolSpec("get_page", "page", {"type": "object", "properties": {}})]
HISTORY = [
    Message("user", "hi"),
    Message("assistant", "", [ToolCall("c1", "get_page", {"page_id": "p"})]),
    Message("tool", '{"ok": 1}', tool_call_id="c1", name="get_page"),
]


async def collect(backend, history=HISTORY):
    return [event async for event in backend.stream("system", history, TOOLS)]


async def test_openai_chat_wire_and_stream():
    seen = []

    def handler(request):
        seen.append(request)
        return sse(
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x1", "function": {"name": "get_", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "page", "arguments": 'ge_id": "p"}'}}]}}]},
            "[DONE]",
        )

    backend = OpenAiChatBackend("openai", "https://api.test/chat", "m", "key")
    backend.transport = httpx.MockTransport(handler)
    events = await collect(backend)
    assert [type(e).__name__ for e in events] == ["ReasoningDelta", "TextDelta", "ToolCallRequest"]
    assert (events[2].id, events[2].name, json.loads(events[2].arguments)) == ("x1", "get_page", {"page_id": "p"})
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer key"
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool"]
    assert body["messages"][2]["tool_calls"][0]["function"] == {"name": "get_page", "arguments": '{"page_id": "p"}'}
    assert body["tools"][0]["function"]["name"] == "get_page" and body["stream"] is True


async def test_openai_chat_quota_is_reported():
    backend = OpenAiChatBackend("openai", "https://api.test/chat", "m", "key")
    backend.transport = httpx.MockTransport(lambda r: httpx.Response(429, text="slow down"))
    from urakatsu.providers.translation.errors import QuotaExceeded

    with pytest.raises(QuotaExceeded):
        await collect(backend)


async def test_anthropic_wire_and_stream():
    seen = []

    def handler(request):
        seen.append(request)
        return sse(
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "get_page"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
            ("message_stop", {"type": "message_stop"}),
        )

    backend = AnthropicBackend("claude-x", "key")
    backend.transport = httpx.MockTransport(handler)
    events = await collect(backend)
    assert [type(e).__name__ for e in events] == ["TextDelta", "ToolCallRequest"]
    body = json.loads(seen[0].content)
    assert body["system"] == "system" and body["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert body["messages"][1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "get_page", "input": {"page_id": "p"}}
    assert body["messages"][2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": '{"ok": 1}'}]}


# -- ChatGPT --------------------------------------------------------------------------------------


def jwt(claims):
    encode = lambda data: base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{encode({'alg': 'none'})}.{encode(claims)}.sig"


ID_TOKEN = jwt({"email": "a@b.c", "https://api.openai.com/auth": {"chatgpt_account_id": "acct", "chatgpt_plan_type": "plus"}})


def signed_in(tmp_path, transport=None, expires_in=3600):
    auth = ChatGptAuth(tmp_path / "chatgpt.json", transport=transport)
    auth._store({"id_token": ID_TOKEN, "access_token": jwt({"exp": time.time() + expires_in}), "refresh_token": "r1"})
    return auth


def test_authorize_url_and_pkce():
    verifier, challenge = pkce_pair()
    assert len(verifier) == 86 and "=" not in challenge
    url = httpx.URL(authorize_url("http://127.0.0.1:1455/auth/callback", challenge, "st"))
    params = dict(url.params)
    assert url.host == "auth.openai.com" and url.path == "/oauth/authorize"
    assert params["client_id"] == CLIENT_ID and params["code_challenge_method"] == "S256"
    assert params["codex_cli_simplified_flow"] == "true" and "offline_access" in params["scope"]


def test_exchange_stores_account(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id_token": ID_TOKEN, "access_token": "a1", "refresh_token": "r1"})

    auth = ChatGptAuth(tmp_path / "chatgpt.json", transport=httpx.MockTransport(handler))
    auth.exchange("code", "http://127.0.0.1:1455/auth/callback", "verifier")
    form = dict(httpx.QueryParams(seen[0].content.decode()))
    assert form["grant_type"] == "authorization_code" and form["code_verifier"] == "verifier"
    status = auth.status()
    assert (status["signed_in"], status["email"], status["plan"]) == (True, "a@b.c", "plus")
    assert jwt_claims(ID_TOKEN)["email"] == "a@b.c"
    auth.logout()
    assert not auth.signed_in()


def test_refresh_when_about_to_expire(tmp_path):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"access_token": jwt({"exp": time.time() + 3600}), "refresh_token": "r2"})

    auth = signed_in(tmp_path, httpx.MockTransport(handler), expires_in=60)
    headers = auth.headers()
    assert seen == [{"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": "r1"}]
    assert headers["ChatGPT-Account-ID"] == "acct" and headers["originator"] == "codex_cli_rs"
    auth.headers()
    assert len(seen) == 1  # fresh now


def test_expired_refresh_signs_out(tmp_path):
    auth = signed_in(tmp_path, httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "invalid_grant"})), expires_in=0)
    with pytest.raises(Exception, match="sign in again"):
        auth.headers()
    assert not auth.signed_in()


async def test_chatgpt_models_and_lite_stream(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "b", "display_name": "B", "visibility": "list", "priority": 2, "use_responses_lite": False},
                        {"slug": "hidden", "visibility": "hide", "priority": 0},
                        {"slug": "a", "display_name": "A", "visibility": "list", "priority": 1, "use_responses_lite": True, "default_reasoning_level": "low", "supports_reasoning_summary_parameter": True},
                    ]
                },
            )
        return sse(
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "f1", "name": "get_page", "arguments": "{}", "namespace": "functions"}},
            {"type": "response.completed", "response": {}},
        )

    auth = signed_in(tmp_path, httpx.MockTransport(handler))
    auth._client_version = "1.0.0"
    models = await auth.models()
    assert [m.id for m in models] == ["chatgpt:a", "chatgpt:b"]
    assert seen[0].url.params["client_version"] == "1.0.0"

    backend = await auth.backend("a")
    assert isinstance(backend, ChatGptBackend) and backend.lite
    backend.transport = httpx.MockTransport(handler)
    events = await collect(backend)
    assert [type(e).__name__ for e in events] == ["TextDelta", "ToolCallRequest"]
    request = seen[-1]
    body = json.loads(request.content)
    assert request.url == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["x-openai-internal-codex-responses-lite"] == "true"
    assert request.headers["chatgpt-account-id"] == "acct"
    assert "tools" not in body and body["store"] is False
    # Responses Lite rejects a request whose reasoning context is not all_turns.
    assert body["reasoning"] == {"effort": "low", "summary": "auto", "context": "all_turns"}
    assert body["input"][0]["type"] == "additional_tools" and body["input"][0]["tools"][0]["name"] == "functions"
    assert body["input"][1] == {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "system"}]}
    assert body["input"][3] == {"type": "function_call", "call_id": "c1", "name": "get_page", "arguments": '{"page_id": "p"}', "namespace": "functions"}
    assert body["input"][4] == {"type": "function_call_output", "call_id": "c1", "output": '{"ok": 1}'}

    plain = await auth.backend("b")
    body = plain.body("s", HISTORY, TOOLS)
    assert "tools" in body and not plain.lite and body["reasoning"] is None


# -- models and routes ----------------------------------------------------------------------------


async def test_model_directory_resolves_provider_backends(workspace, tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LM_STUDIO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    directory = ModelDirectory(workspace, ChatGptAuth(tmp_path / "none.json"))
    with pytest.raises(ValueError, match="no API key"):
        await directory.backend("openai:gpt-x")
    with pytest.raises(ValueError, match="cannot run an agent"):
        await directory.backend("deepl:x")
    with pytest.raises(ValueError, match="sign in"):
        await directory.backend("chatgpt:x")
    workspace.config.providers = {"claude": {"api_key": "k"}, "lm-studio": {"base_url": "http://box:1"}}
    claude = await directory.backend("claude:opus")
    assert isinstance(claude, AnthropicBackend) and claude.api_key == "k"
    studio = await directory.backend("lm-studio:qwen")
    assert studio.url == "http://box:1/v1/chat/completions" and studio.api_key is None


async def test_routes(workspace, project):
    project_id, page_id = project
    app = build_app(workspace, UrakatsuServicer(workspace), serve_ui=False)
    agents = app.state.agents
    backend = ScriptedBackend([[TextDelta("Hello")]])

    async def resolve(ref):
        return backend

    agents.models.backend = resolve
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/v1/agent/agents")).json()[0]["id"] == "assistant"
        assert (await client.get("/api/v1/agent/auth/chatgpt")).json()["signed_in"] is False
        response = await client.post("/api/v1/agent/messages", json={"text": "hi", "model": "test:model", "project_id": project_id, "page_id": page_id})
        conversation_id = response.json()["conversation_id"]
        stream = await client.get(f"/api/v1/agent/conversations/{conversation_id}/events")
        events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
        assert events[0] == {"type": "user", "content": "hi"} and events[-1]["type"] == "done"
        assert (await client.get(f"/api/v1/agent/conversations?project_id={project_id}")).json()[0]["id"] == conversation_id
        assert (await client.post("/api/v1/agent/messages", json={"text": " ", "model": "test:model"})).status_code == 400
        assert (await client.delete(f"/api/v1/agent/conversations/{conversation_id}")).status_code == 204
        assert (await client.get(f"/api/v1/agent/conversations/{conversation_id}")).status_code == 404


def test_store_rejects_odd_ids(tmp_path):
    store = ConversationStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("../secrets")
    assert Turns(store).running("x") is False


async def test_model_list_is_saved_and_marked_stale(workspace, tmp_path, monkeypatch):
    calls = []

    def listing():
        calls.append(1)
        return [{"provider": "deepseek", "provider_name": "DeepSeek", "model": "chat", "name": "Chat"}, {"provider": "deepl", "provider_name": "DeepL", "model": None, "name": "DeepL"}]

    monkeypatch.setattr(workspace, "translation_models", listing)
    path = tmp_path / "models.json"
    directory = ModelDirectory(workspace, ChatGptAuth(tmp_path / "none.json"), path)
    models, stale = await directory.list()
    assert [m.id for m in models] == ["deepseek:chat"] and not stale and len(calls) == 1

    # A new process answers from the saved listing without listing again.
    again = ModelDirectory(workspace, ChatGptAuth(tmp_path / "none.json"), path)
    models, stale = await again.list()
    assert [m.id for m in models] == ["deepseek:chat"] and not stale and len(calls) == 1

    # Changing provider settings marks it stale; a refresh lists again.
    workspace.config.providers = {"deepseek": {"api_key": "k"}}
    assert (await again.list())[1] is True
    models, stale = await again.list(refresh=True)
    assert not stale and len(calls) == 2


async def test_concurrent_refreshes_share_one_listing(workspace, tmp_path, monkeypatch):
    calls = []

    def listing():
        calls.append(1)
        time.sleep(0.05)
        return []

    monkeypatch.setattr(workspace, "translation_models", listing)
    directory = ModelDirectory(workspace, ChatGptAuth(tmp_path / "none.json"), tmp_path / "models.json")
    await asyncio.gather(directory.list(refresh=True), directory.list(refresh=True))
    assert len(calls) == 1


def test_listing_a_local_server_connects_with_a_short_timeout(monkeypatch):
    from urakatsu.providers.translation.remote import base

    seen = []

    class Client:
        def __init__(self, transport=None, timeout=None):
            seen.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, *args, **kwargs):
            return httpx.Response(200, json={})

    monkeypatch.setattr(base.httpx, "Client", Client)
    base.send_json_sync("x", "GET", "http://localhost:1234/v1/models")
    base.send_json_sync("x", "GET", "https://api.example.com/v1/models")
    assert seen[0].connect == base.LOOPBACK_CONNECT_TIMEOUT and seen[1].connect == base.LISTING_TIMEOUT


# -- image edits ----------------------------------------------------------------------------------


def png_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_window_adds_context_and_takes_the_output_shape():
    region = Box(500, 800, 600, 860)
    window, size = plan_window(region, (1200, 1800))
    assert size == (1536, 1024)
    assert window.left <= 500 and window.top <= 800 and window.right >= 600 and window.bottom >= 860
    assert abs(window.width / window.height - 1.5) < 0.01
    # At the page edge the window stays on the page and padding makes up the shape.
    window, size = plan_window(Box(0, 0, 80, 1700), (100, 1800))
    assert (window.left, window.right) == (0, 100) and size == (1024, 1536)
    padded, offset = pad_to(Image.new("RGB", (window.width, window.height)), size)
    assert abs(padded.width / padded.height - 1024 / 1536) < 0.01 and offset[1] == 0
    # The whole page is sent without a margin.
    assert plan_window(Box(0, 0, 1000, 1500), (1000, 1500), margin=False) == (Box(0, 0, 1000, 1500), (1024, 1536))


class SolidEditor(ImageEditor):
    """Answers every edit with a solid colour at the asked size, and records the requests."""

    def __init__(self, colour=(255, 0, 0)):
        self.colour = colour
        self.requests = []

    async def edit(self, image, prompt, size, quality="auto"):
        self.requests.append((image.size, prompt, size, quality))
        return Image.new("RGB", size, self.colour)


async def test_edit_region_keeps_only_the_region():
    art = Image.new("RGB", (400, 300), (255, 255, 255))
    layer, window, size = await edit_region(SolidEditor(), art, Box(150, 100, 250, 160), "make it red")
    assert layer.size == (400, 300) and layer.mode == "RGBA"
    assert layer.getpixel((200, 130)) == (255, 0, 0, 255)  # inside
    assert layer.getpixel((152, 102))[3] == 255  # still inside, near the edge
    assert layer.getpixel((5, 5))[3] == 0 and layer.getpixel((window.right - 1, window.bottom - 1))[3] == 0
    page, _, _ = await edit_region(SolidEditor(), art, None, "all of it")
    assert page.getpixel((0, 0)) == (255, 0, 0, 255) and page.getpixel((399, 299)) == (255, 0, 0, 255)


async def test_chatgpt_image_editor_request_refresh_and_retries(tmp_path, monkeypatch):
    from urakatsu.agents import image_edit

    monkeypatch.setattr(image_edit, "RETRY_DELAY", 0.0)
    seen, answers = [], []

    def handler(request):
        seen.append(request)
        return answers.pop(0)

    ok = httpx.Response(200, json={"created": 1, "data": [{"b64_json": png_b64(Image.new("RGB", (1024, 1024), "blue"))}]})
    answers += [httpx.Response(401, text="expired"), httpx.Response(503, text="busy"), ok]
    auth = signed_in(tmp_path)
    monkeypatch.setattr(auth, "headers", lambda force_refresh=False: {"Authorization": "Bearer t", "forced": str(force_refresh)})
    editor = ChatGptImageEditor(auth, transport=httpx.MockTransport(handler))
    result = await editor.edit(Image.new("RGB", (300, 300), "white"), "translate", (1024, 1024), "high")
    assert result.size == (1024, 1024) and result.getpixel((0, 0)) == (0, 0, 255)
    assert str(seen[0].url).endswith("/backend-api/codex/images/edits")
    body = json.loads(seen[0].content)
    assert (body["model"], body["size"], body["quality"], body["prompt"]) == ("gpt-image-2", "1024x1024", "high", "translate")
    assert body["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert [r.headers["forced"] for r in seen] == ["False", "True", "True"]

    answers[:] = [httpx.Response(429, text='{"error": {"type": "usage_limit_reached"}}')]
    with pytest.raises(QuotaExceeded):
        await editor.edit(Image.new("RGB", (8, 8)), "x", (1024, 1024))
    assert not answers  # a spent image limit is not retried


async def test_edit_image_tool_adds_an_undoable_edit_layer(workspace, project):
    project_id, page_id = project
    first, second = [b.layer_id for b in workspace.page(project_id, page_id).blocks]
    editor = SolidEditor()
    tools = get_agent("assistant").toolset()
    context = ToolContext(workspace, project_id, page_id, image_editor=editor)

    ok, text = await tools.call(context, "edit_image", {"instruction": "Translate the SFX into English.", "layer_ids": [second]})
    assert ok, text
    result = json.loads(text)
    assert result["hidden_text_layers"] == [second] and result["region"] == {"x": 60, "y": 60, "width": 40, "height": 20}
    assert "crop of a manga page" in editor.requests[0][1] and "Translate the SFX" in editor.requests[0][1]
    page = workspace.page(project_id, page_id)
    [raster] = [r for r in page.raster_layers if r.id == result["layer_id"]]
    assert raster.kind == "edit" and raster.name.startswith("Edit: Translate")
    assert [b.visible for b in page.blocks] == [True, False]
    data, _, _, _ = workspace.asset(project_id, page_id, f"layer:{raster.id}")
    layer = Image.open(io.BytesIO(data))
    assert layer.getpixel((80, 70)) == (255, 0, 0, 255) and layer.getpixel((300, 250))[3] == 0
    # The render shows the edit; one undo removes it and shows the text block again.
    render = Image.open(io.BytesIO(workspace.render(project_id, page_id))).convert("RGB")
    assert render.getpixel((80, 70)) == (255, 0, 0)
    workspace.undo(project_id)
    page = workspace.page(project_id, page_id)
    assert not any(r.kind == "edit" for r in page.raster_layers) and all(b.visible for b in page.blocks)

    ok, text = await tools.call(context, "edit_image", {"instruction": "Translate the page into English.", "mode": "page", "quality": "high"})
    assert ok, text
    assert json.loads(text)["hidden_text_layers"] == [first, second] and editor.requests[-1][2:] == ((1536, 1024), "high")

    ok, text = await tools.call(context, "edit_image", {"instruction": "x", "region": {"x": 900, "y": 900, "width": 10, "height": 10}})
    assert not ok and "outside the page" in text


class DeletingEditor(SolidEditor):
    """Deletes a text block while the image model works, as a re-run in the meantime would."""

    def __init__(self, workspace, project_id, page_id, layer_id):
        super().__init__()
        self.target = (workspace, project_id, page_id, layer_id)

    async def edit(self, image, prompt, size, quality="auto"):
        workspace, project_id, page_id, layer_id = self.target
        workspace.delete_text_block(project_id, page_id, layer_id)
        return await super().edit(image, prompt, size, quality)


async def test_edit_image_hides_the_text_layers_there_are_when_it_commits(workspace, project):
    project_id, page_id = project
    first, second = [b.layer_id for b in workspace.page(project_id, page_id).blocks]
    tools = get_agent("assistant").toolset()

    context = ToolContext(workspace, project_id, page_id, image_editor=DeletingEditor(workspace, project_id, page_id, second))
    ok, text = await tools.call(context, "edit_image", {"instruction": "x", "layer_ids": [second]})
    assert ok, text
    assert json.loads(text)["hidden_text_layers"] == []

    created = workspace.create_text_block(project_id, page_id, {"x": 200, "y": 200, "width": 30, "height": 30}).blocks[-1].layer_id
    context = ToolContext(workspace, project_id, page_id, image_editor=DeletingEditor(workspace, project_id, page_id, first))
    ok, text = await tools.call(context, "edit_image", {"instruction": "x", "mode": "page"})
    assert ok, text
    assert json.loads(text)["hidden_text_layers"] == [created]


def test_command_parsing():
    command, rest = parse_command("/translate-sfx make it louder")
    assert (command.name, command.tool, rest) == ("translate-sfx", "edit_image", "make it louder")
    assert parse_command("/edit-image")[1] == "" and parse_command("hello /edit-image") is None
    with pytest.raises(ValueError, match="unknown command /nope"):
        parse_command("/nope")


async def test_slash_command_runs_the_tool_without_a_model(workspace, project, monkeypatch):
    from urakatsu.agents import service as agent_service

    project_id, page_id = project
    first, second = [b.layer_id for b in workspace.page(project_id, page_id).blocks]
    editor = SolidEditor()
    monkeypatch.setattr(agent_service, "ChatGptImageEditor", lambda auth: editor)
    agents = AgentService(workspace)

    async def no_model(ref):
        raise AssertionError("a command needs no model")

    agents.models.backend = no_model
    with pytest.raises(ValueError, match="select the sound effect"):
        await agents.send(None, "/translate-sfx", "test:model", project_id=project_id, page_id=page_id)
    assert agents.store.list(project_id) == []  # nothing was created

    conversation_id = await agents.send(None, "/translate-sfx keep it red", "test:model", project_id=project_id, page_id=page_id, selection=[second])
    events = [event async for event in agents.turns.events(conversation_id)]
    assert [e["type"] for e in events] == ["user", "tool_call", "tool_result", "text", "done"]
    assert events[1]["arguments"]["layer_ids"] == [second] and events[2]["ok"] and events[-1]["status"] == "done"
    prompt = editor.requests[0][1]
    assert "into English" in prompt and "Also: keep it red" in prompt
    messages = agents.conversation(conversation_id)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == "/translate-sfx keep it red" and "image edit layer" in messages[-1]["content"]
    assert [b.visible for b in workspace.page(project_id, page_id).blocks] == [True, False]

