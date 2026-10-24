"""Sign in with ChatGPT, and chat through the signed-in subscription.

This follows OpenAI's Codex CLI (github.com/openai/codex, ``codex-rs``), the first-party
client that offers this sign-in; the constants below are read from its source:

- ``login/src/server.rs``: OAuth authorization code flow with PKCE (S256, 64-byte
  verifier) against ``auth.openai.com`` with Codex's public client id. The redirect must
  be ``http://127.0.0.1:1455/auth/callback`` (or port 1457): those are the only ones the
  client id allows, so the callback server binds one of them.
- ``login/src/auth/manager.rs``: tokens are refreshed (JSON body) when the access token
  expires within five minutes, or after a 401.
- ``login/src/token_data.rs``: the account id, email and plan are claims in the id token.
- ``core/src/client.rs``: inference is the Responses API at
  ``chatgpt.com/backend-api/codex/responses``, streamed, with ``store: false``. Models
  marked ``use_responses_lite`` take their tools as an ``additional_tools`` input item in
  the ``functions`` namespace plus the ``x-openai-internal-codex-responses-lite`` header,
  instead of the ``tools`` field, and need ``reasoning.context`` set to ``all_turns``
  (``build_reasoning``). The reasoning summary is requested only from models that
  declare ``supports_reasoning_summary_parameter``.
- ``codex-api/src/endpoint/models.rs``: the account's models come from ``/models``.

Tokens are stored in ``~/.urakatsu/auth/chatgpt.json``. This is not an official API for
third-party apps; if OpenAI changes it, sign-in or chat here stops working until this
module follows.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx

from urakatsu import __version__
from urakatsu.core.config import home

from .backends.base import AgentError, ChatBackend, sse_events
from .messages import BackendEvent, Message, ReasoningDelta, TextDelta, ToolCallRequest, ToolSpec

log = logging.getLogger(__name__)

ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
ORIGINATOR = "codex_cli_rs"
PORTS = (1455, 1457)
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
# Used only when GitHub cannot be reached: the newest Codex release when this was written.
FALLBACK_CLIENT_VERSION = "0.164.0"
# How long a looked-up Codex version is reused (it is saved next to the tokens).
CLIENT_VERSION_TTL = 24 * 3600
REFRESH_WINDOW = 5 * 60
LOGIN_TIMEOUT = 10 * 60
AUTH_CLAIMS = "https://api.openai.com/auth"
PROFILE_CLAIMS = "https://api.openai.com/profile"
NAMESPACE = "functions"
LITE_HEADER = "x-openai-internal-codex-responses-lite"
# ``response.failed`` error codes that may pass if the step is sent again.
TRANSIENT_CODES = ("server_error", "server_is_overloaded", "rate_limit_exceeded", "slow_down")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def jwt_claims(token: Optional[str]) -> dict[str, Any]:
    """A JWT's payload, unverified (it came straight from the token endpoint over TLS)."""
    try:
        payload = (token or "").split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}


def authorize_url(redirect_uri: str, challenge: str, state: str) -> str:
    query = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "scope": SCOPE,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{ISSUER}/oauth/authorize?{urllib.parse.urlencode(query)}"


class ChatGptAuth:
    """The signed-in ChatGPT account: login, token storage and refresh, model listing."""

    def __init__(self, path: Optional[Path] = None, transport: Optional[httpx.BaseTransport] = None):
        self.path = path or home() / "auth" / "chatgpt.json"
        self.transport = transport
        self.async_transport: Optional[httpx.AsyncBaseTransport] = None
        self.last_error: Optional[str] = None
        self._login: Optional[_LoginServer] = None
        self._client_version: Optional[str] = None
        self._models: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- stored tokens ---------------------------------------------------------------------------

    def _read(self) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, tokens: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(tokens), encoding="utf-8")
        temporary.replace(self.path)

    def signed_in(self) -> bool:
        tokens = self._read()
        return bool(tokens and tokens.get("access_token") and tokens.get("refresh_token"))

    def status(self) -> dict[str, Any]:
        tokens = self._read() or {}
        claims = jwt_claims(tokens.get("id_token"))
        auth = claims.get(AUTH_CLAIMS) or {}
        return {
            "signed_in": self.signed_in(),
            "email": claims.get("email") or (claims.get(PROFILE_CLAIMS) or {}).get("email"),
            "plan": auth.get("chatgpt_plan_type"),
            "pending": self._login is not None and self._login.running(),
            "login_url": self._login.url if self._login is not None and self._login.running() else None,
            "error": self.last_error,
        }

    def logout(self) -> None:
        self.path.unlink(missing_ok=True)
        self._models.clear()

    def _store(self, response: dict[str, Any], previous: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        tokens = dict(previous or {})
        for key in ("id_token", "access_token", "refresh_token"):
            if response.get(key):
                tokens[key] = response[key]
        claims = jwt_claims(tokens.get("id_token"))
        tokens["account_id"] = (claims.get(AUTH_CLAIMS) or {}).get("chatgpt_account_id") or tokens.get("account_id")
        tokens["last_refresh"] = time.time()
        self._write(tokens)
        return tokens

    # -- login -----------------------------------------------------------------------------------

    def start_login(self, open_browser: bool = True) -> str:
        """Starts the callback server and returns the sign-in URL (also opened in the browser)."""
        if self._login is not None and self._login.running():
            url = self._login.url
        else:
            self.last_error = None
            self._login = _LoginServer(self)
            url = self._login.start()
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # the UI shows the link as well
                pass
        return url

    def cancel_login(self) -> None:
        if self._login is not None:
            self._login.stop()
            self._login = None

    def exchange(self, code: str, redirect_uri: str, verifier: str) -> None:
        form = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
        with httpx.Client(transport=self.transport, timeout=30.0) as client:
            response = client.post(f"{ISSUER}/oauth/token", data=form)
        if response.status_code >= 400:
            raise AgentError(f"ChatGPT sign-in failed: HTTP {response.status_code}: {response.text[:300]}")
        self._store(response.json())
        self._models.clear()

    # -- access ----------------------------------------------------------------------------------

    def _refresh(self, tokens: dict[str, Any]) -> dict[str, Any]:
        body = {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": tokens["refresh_token"]}
        with httpx.Client(transport=self.transport, timeout=30.0) as client:
            response = client.post(f"{ISSUER}/oauth/token", json=body)
        if response.status_code >= 400:
            text = response.text
            if response.status_code in (400, 401) and any(
                marker in text for marker in ("invalid_grant", "refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated")
            ):
                self.logout()
                raise AgentError("your ChatGPT session expired; sign in again")
            raise AgentError(f"ChatGPT token refresh failed: HTTP {response.status_code}: {text[:300]}")
        return self._store(response.json(), tokens)

    def credentials(self, force_refresh: bool = False) -> tuple[str, str]:
        """A current access token and the account id, refreshing first when it is about to expire."""
        with self._lock:
            tokens = self._read()
            if not tokens or not tokens.get("access_token"):
                raise AgentError("sign in with ChatGPT first")
            expires = jwt_claims(tokens["access_token"]).get("exp")
            stale = expires is None and time.time() - tokens.get("last_refresh", 0) > 8 * 86400
            if force_refresh or stale or (expires is not None and expires - time.time() < REFRESH_WINDOW):
                tokens = self._refresh(tokens)
            return tokens["access_token"], tokens.get("account_id") or ""

    def headers(self, force_refresh: bool = False) -> dict[str, str]:
        token, account = self.credentials(force_refresh)
        headers = {
            "Authorization": f"Bearer {token}",
            "originator": ORIGINATOR,
            "User-Agent": f"urakatsu/{__version__}",
        }
        if account:
            headers["ChatGPT-Account-ID"] = account
        return headers

    # -- models ----------------------------------------------------------------------------------

    def client_version(self) -> str:
        """The newest Codex release: ``/models`` hides models newer than the client asking.

        Looked up on GitHub at most once a day; the answer is saved beside the tokens.
        """
        if self._client_version is not None:
            return self._client_version
        path = self.path.with_name("codex-version.json")
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - saved["at"] < CLIENT_VERSION_TTL:
                self._client_version = str(saved["version"])
                return self._client_version
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            with httpx.Client(transport=self.transport, timeout=5.0) as client:
                tag = client.get(CODEX_RELEASES).raise_for_status().json()["tag_name"]
            self._client_version = tag.removeprefix("rust-v")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"at": time.time(), "version": self._client_version}), encoding="utf-8")
        except Exception:
            self._client_version = FALLBACK_CLIENT_VERSION
        return self._client_version

    async def models(self) -> list[Any]:
        from .models import AgentModel

        def listing() -> list[dict[str, Any]]:
            url = f"{CODEX_BASE_URL}/models"
            params = {"client_version": self.client_version()}
            with httpx.Client(transport=self.transport, timeout=15.0) as client:
                response = client.get(url, params=params, headers=self.headers())
                if response.status_code == 401:
                    response = client.get(url, params=params, headers=self.headers(force_refresh=True))
            if response.status_code >= 400:
                raise AgentError(f"ChatGPT model list failed: HTTP {response.status_code}: {response.text[:300]}")
            return response.json().get("models") or []

        entries = await asyncio.to_thread(listing)
        self._models = {entry["slug"]: entry for entry in entries if entry.get("slug")}
        shown = [e for e in entries if e.get("slug") and e.get("visibility", "list") == "list"]
        shown.sort(key=lambda e: e.get("priority", 1_000))
        return [
            AgentModel(
                id=f"chatgpt:{e['slug']}",
                provider="chatgpt",
                provider_name="ChatGPT",
                model=e["slug"],
                name=e.get("display_name") or e["slug"],
                source="chatgpt",
            )
            for e in shown
        ]

    async def backend(self, model: str) -> "ChatGptBackend":
        if not self.signed_in():
            raise ValueError("sign in with ChatGPT to use its models")
        if model not in self._models:
            await self.models()
        return ChatGptBackend(self, model, self._models.get(model, {}))


class _LoginServer:
    """The localhost callback the sign-in page redirects to; one sign-in at a time."""

    def __init__(self, auth: ChatGptAuth):
        self.auth = auth
        self.verifier, self.challenge = pkce_pair()
        self.state = _b64url(secrets.token_bytes(32))
        self.url = ""
        self.redirect_uri = ""
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.deadline = time.monotonic() + LOGIN_TIMEOUT

    def start(self) -> str:
        login = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                if parsed.path == "/cancel":
                    self._reply(200, "Sign-in cancelled.")
                    login.finish("sign-in was cancelled")
                    return
                if parsed.path != "/auth/callback":
                    self._reply(404, "Not found.")
                    return
                if query.get("state", [""])[0] != login.state:
                    self._reply(400, "This sign-in link is out of date. Start again from Urakatsu.")
                    return
                if "error" in query:
                    message = query.get("error_description", query["error"])[0]
                    self._reply(400, f"Sign-in failed: {message}")
                    login.finish(message)
                    return
                try:
                    login.auth.exchange(query.get("code", [""])[0], login.redirect_uri, login.verifier)
                except Exception as error:
                    self._reply(500, f"Sign-in failed: {error}")
                    login.finish(str(error))
                    return
                self._reply(200, "Signed in to ChatGPT. You can close this tab and return to Urakatsu.")
                login.finish(None)

            def _reply(self, status: int, text: str) -> None:
                body = (
                    "<!doctype html><meta charset=utf-8><title>Urakatsu</title>"
                    "<body style='font:15px system-ui;display:grid;place-items:center;height:90vh'>"
                    f"<p>{text}</p></body>"
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        for port in PORTS:
            _cancel_previous(port)
            try:
                self.server = HTTPServer(("127.0.0.1", port), Handler)
                break
            except OSError:
                continue
        if self.server is None:
            raise AgentError("ports 1455 and 1457 are busy (is another ChatGPT sign-in, such as Codex, open?)")
        self.server.timeout = 0.5
        self.redirect_uri = f"http://127.0.0.1:{self.server.server_port}/auth/callback"
        self.url = authorize_url(self.redirect_uri, self.challenge, self.state)
        self.thread = threading.Thread(target=self._serve, name="chatgpt-login", daemon=True)
        self.thread.start()
        return self.url

    def _serve(self) -> None:
        server = self.server
        while server is not None and self.server is server and time.monotonic() < self.deadline:
            server.handle_request()
        if self.server is server:
            self.finish("sign-in timed out")

    def running(self) -> bool:
        return self.server is not None

    def finish(self, error: Optional[str]) -> None:
        self.auth.last_error = error
        self.stop()

    def stop(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            server.server_close()


def _cancel_previous(port: int) -> None:
    """Asks a sign-in server left on ``port`` (ours or Codex's) to shut down, as Codex does."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3) as connection:
            connection.sendall(f"GET /cancel HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode("ascii"))
            connection.recv(64)
        time.sleep(0.2)
    except OSError:
        pass


# -- inference ----------------------------------------------------------------------------------


def wire_input(system: str, messages: list[Message], tools: list[ToolSpec], lite: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if lite and tools:
        items.append({"type": "additional_tools", "role": "developer", "tools": [_namespace(tools)]})
    items.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": system}]})
    for message in messages:
        if message.role == "user":
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": message.content}]})
        elif message.role == "assistant":
            if message.content:
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": message.content}]})
            for call in message.tool_calls:
                item = {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments_json()}
                if lite:
                    item["namespace"] = NAMESPACE
                items.append(item)
        else:
            items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content})
    return items


def _function(tool: ToolSpec) -> dict[str, Any]:
    return {"type": "function", "name": tool.name, "description": tool.description, "strict": False, "parameters": tool.parameters}


def _namespace(tools: list[ToolSpec]) -> dict[str, Any]:
    return {"type": "namespace", "name": NAMESPACE, "description": "", "tools": [_function(t) for t in tools]}


class ChatGptBackend(ChatBackend):
    def __init__(self, auth: ChatGptAuth, model: str, info: Optional[dict[str, Any]] = None):
        """``info`` is the model's ``/models`` entry, which decides the request shape."""
        self.auth = auth
        self.model = model
        self.info = info or {}
        self.lite = bool(self.info.get("use_responses_lite"))
        self.ref = f"chatgpt:{model}"
        # Codex budgets a share of the window (``effective_context_window_percent``) and so do we.
        window = self.info.get("context_window")
        if isinstance(window, int) and window > 0:
            self.context_window = window * int(self.info.get("effective_context_window_percent") or 100) // 100

    def reasoning(self) -> Optional[dict[str, Any]]:
        reasoning: dict[str, Any] = {}
        if self.info.get("default_reasoning_level"):
            reasoning["effort"] = self.info["default_reasoning_level"]
        if self.info.get("supports_reasoning_summary_parameter"):
            reasoning["summary"] = "auto"
        if self.lite:
            reasoning["context"] = "all_turns"
        return reasoning or None

    def body(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "input": wire_input(system, messages, tools, self.lite),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": self.reasoning(),
            "store": False,
            "include": [],
        }
        if tools and not self.lite:
            body["tools"] = [_function(t) for t in tools]
        return body

    async def stream(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[BackendEvent]:
        body = self.body(system, messages, tools)
        url = f"{CODEX_BASE_URL}/responses"
        for attempt in range(2):
            headers = await asyncio.to_thread(self.auth.headers, attempt > 0)
            headers["Accept"] = "text/event-stream"
            if self.lite:
                headers[LITE_HEADER] = "true"
            try:
                async for event in self._events(url, headers, body):
                    yield event
                return
            except _Unauthorized:
                if attempt:
                    raise AgentError("ChatGPT rejected the session; sign in again") from None

    async def _events(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> AsyncIterator[BackendEvent]:
        from urakatsu.providers.translation.errors import ApiError

        try:
            async for event, data in sse_events("chatgpt", url, headers=headers, body=body, transport=self.transport):
                kind = data.get("type", event)
                if kind == "response.output_text.delta":
                    yield TextDelta(data.get("delta", ""))
                elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    yield ReasoningDelta(data.get("delta", ""))
                elif kind == "response.output_item.done":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        yield ToolCallRequest(item.get("call_id") or item.get("id") or "", item.get("name", ""), item.get("arguments") or "")
                elif kind in ("response.failed", "error"):
                    error = (data.get("response") or {}).get("error") or data.get("error") or data
                    code = error.get("code") if isinstance(error, dict) else None
                    raise AgentError(
                        f"ChatGPT: {error.get('message') if isinstance(error, dict) else error}",
                        transient=code in TRANSIENT_CODES,
                        overflow=code == "context_length_exceeded",
                    )
                elif kind == "response.incomplete":
                    reason = ((data.get("response") or {}).get("incomplete_details") or {}).get("reason")
                    if reason == "content_filter":
                        raise AgentError("ChatGPT stopped the reply (content filter)")
                elif kind == "response.completed":
                    return
        except ApiError as error:
            if error.status == 401:
                raise _Unauthorized() from error
            raise


class _Unauthorized(Exception):
    pass
