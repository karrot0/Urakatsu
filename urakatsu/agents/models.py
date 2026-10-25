"""The models an agent can run on, and the backend for each.

Three sources, none of them a hand-written model list:
- ChatGPT: the signed-in account's models (``chatgpt.py``).
- Translation providers that are LLMs: whatever their listing returns with the keys and
  endpoints set under Settings, Providers. Services that only translate (DeepL, Google
  Cloud, Caiyun) are left out.
- The local translation model: the GGUF model the translation stage runs, on its own
  ``llama-server`` (loaded on first use). It shares that server with translation, so it
  has the translator's context window (``local.N_CTX`` tokens).

A model is addressed as ``provider:model``.

Listing asks every provider over the network (one to several seconds), so the last
listing is saved to ``~/.urakatsu/agents/models.json`` and answered at once. It is marked
stale after ``LISTING_TTL`` seconds, or as soon as the provider settings or the ChatGPT
sign-in change; the UI then asks for a fresh listing in the background.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from urakatsu.core.config import home
from urakatsu.core.stage import Stage
from urakatsu.providers.translation.remote import lm_studio, openai_compatible
from urakatsu.providers.translation.remote.base import credentials, key_ring
from urakatsu.providers.translation.remote.claude import ENV_KEY as CLAUDE_ENV
from urakatsu.providers.translation.remote.deepseek import ENV_KEY as DEEPSEEK_ENV, URL as DEEPSEEK_URL
from urakatsu.providers.translation.remote.gemini import ENV_KEY as GEMINI_ENV
from urakatsu.providers.translation.remote.grok import ENV_KEY as XAI_ENV
from urakatsu.providers.translation.remote.minimax import ENV_KEY as MINIMAX_ENV, URL as MINIMAX_URL
from urakatsu.providers.translation.remote.openai import URL as OPENAI_URL
from urakatsu.providers.translation.remote.openrouter import CHAT_URL as OPENROUTER_URL, ENV_KEY as OPENROUTER_ENV

from .backends.anthropic import AnthropicBackend
from .backends.base import AgentError, ChatBackend
from .backends.openai_chat import OpenAiChatBackend

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.service.workspace import Workspace

    from .chatgpt import ChatGptAuth

XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
LISTING_TTL = 300.0

log = logging.getLogger(__name__)

# Chat-capable translation providers: (environment variable for the key, key required).
CHAT_PROVIDERS: dict[str, tuple[str, bool]] = {
    "openai": ("OPENAI_API_KEY", True),
    "openrouter": (OPENROUTER_ENV, True),
    "deepseek": (DEEPSEEK_ENV, True),
    "grok": (XAI_ENV, True),
    "minimax": (MINIMAX_ENV, True),
    "gemini": (GEMINI_ENV, True),
    "claude": (CLAUDE_ENV, True),
    "lm-studio": (lm_studio.ENV_KEY, False),
    "openai-compatible": (openai_compatible.ENV_KEY, False),
}


class AgentModel(BaseModel):
    id: str  # provider:model
    provider: str
    provider_name: str
    model: str
    name: str
    source: str  # chatgpt | provider | local


def split_ref(ref: str) -> tuple[str, str]:
    provider, sep, model = ref.partition(":")
    if not sep or not provider or not model:
        raise ValueError(f"model {ref!r} is not provider:model")
    return provider, model


class ModelDirectory:
    def __init__(self, workspace: "Workspace", chatgpt: "ChatGptAuth", path: Optional[Path] = None):
        self.workspace = workspace
        self.chatgpt = chatgpt
        self.path = path or home() / "agents" / "models.json"
        self._listing: Optional[dict[str, Any]] = self._read()
        self._fetching: Optional[asyncio.Future] = None

    async def list(self, refresh: bool = False) -> tuple[list[AgentModel], bool]:
        """The models and whether they are stale. Only a refresh, or the first listing ever, waits for the network."""
        listing = self._listing
        if refresh or listing is None:
            return await self._refresh(), False
        stale = time.time() - listing["at"] > LISTING_TTL or listing["key"] != self._key()
        return [AgentModel(**m) for m in listing["models"]], stale

    def _key(self) -> str:
        """What the listing depends on: provider settings, the translation model, the ChatGPT sign-in."""
        config = self.workspace.config
        facts = [config.providers, config.translation.model.model_dump(mode="json"), self.chatgpt.signed_in()]
        return hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()

    async def _refresh(self) -> list[AgentModel]:
        # Concurrent callers share one listing.
        if self._fetching is None or self._fetching.done():
            self._fetching = asyncio.ensure_future(self._fetch())
        return await asyncio.shield(self._fetching)

    async def _fetch(self) -> list[AgentModel]:
        key = self._key()

        async def chatgpt() -> list[AgentModel]:
            if not self.chatgpt.signed_in():
                return []
            try:
                return await self.chatgpt.models()
            except Exception as error:  # a failed listing hides only ChatGPT
                self.chatgpt.last_error = str(error)
                return []

        found, translation = await asyncio.gather(chatgpt(), asyncio.to_thread(self.workspace.translation_models))
        for model in translation:
            if model["provider"] in CHAT_PROVIDERS and model.get("model"):
                found.append(
                    AgentModel(
                        id=f"{model['provider']}:{model['model']}",
                        provider=model["provider"],
                        provider_name=model["provider_name"],
                        model=model["model"],
                        name=model["name"],
                        source="provider",
                    )
                )
        local = self.local_model()
        if local is not None:
            found.append(local)
        self._listing = {"at": time.time(), "key": key, "models": [m.model_dump() for m in found]}
        self._write()
        return found

    def _read(self) -> Optional[dict[str, Any]]:
        try:
            listing = json.loads(self.path.read_text(encoding="utf-8"))
            return listing if {"at", "key", "models"} <= listing.keys() else None
        except (OSError, ValueError, AttributeError):
            return None

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._listing), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as error:
            log.warning("could not save the agent model list: %s", error)

    def local_model(self) -> Optional[AgentModel]:
        selection = self.workspace.config.translation.model
        if selection.provider != "local" or not selection.model:
            return None
        from urakatsu.providers.translation.local import descriptor_for

        try:
            descriptor = descriptor_for(selection.model)
        except ValueError:
            return None
        return AgentModel(
            id=f"local:{selection.model}",
            provider="local",
            provider_name="Local",
            model=selection.model,
            name=descriptor.name,
            source="local",
        )

    async def backend(self, ref: str) -> ChatBackend:
        provider, model = split_ref(ref)
        if provider == "chatgpt":
            return await self.chatgpt.backend(model)
        if provider == "local":
            return await self._local(model, ref)
        if provider not in CHAT_PROVIDERS:
            raise ValueError(f"{provider} cannot run an agent; choose a chat model")
        settings = self.workspace.config.providers.get(provider, {})
        env, required = CHAT_PROVIDERS[provider]
        keys = credentials(settings, "api_key", env)
        if not keys and required:
            raise ValueError(f"{provider} has no API key; add one under Settings, Providers")
        # Same rotation turn as translation: the next ready key.
        key = key_ring(provider).order(keys)[0] if keys else None
        if provider == "claude":
            return AnthropicBackend(model, key or "", ref)
        url = {
            "openai": OPENAI_URL,
            "openrouter": OPENROUTER_URL,
            "deepseek": DEEPSEEK_URL,
            "grok": XAI_CHAT_URL,
            "minimax": MINIMAX_URL,
            "gemini": GEMINI_CHAT_URL,
        }.get(provider)
        if provider == "lm-studio":
            url = lm_studio.endpoint(settings.get("base_url"), "v1/chat/completions")
        elif provider == "openai-compatible":
            url = openai_compatible.endpoint(settings.get("base_url"), "chat/completions")
        assert url is not None
        return OpenAiChatBackend(provider, url, model, key, ref)

    async def _local(self, model: str, ref: str) -> ChatBackend:
        selection = self.workspace.config.translation.model
        if selection.provider != "local" or selection.model != model:
            raise ValueError("only the local model the translation stage uses can run the agent; select it under Settings, Translation")
        provider: Any = self.workspace.pipeline.provider(Stage.TRANSLATION)
        await provider.load()
        self.workspace.pipeline.touch()  # an agent turn uses the model too; idle unload reloads it next turn
        server = getattr(provider, "_server", None)
        if server is None:
            raise AgentError("the local model did not start")
        from urakatsu.providers.translation.local import N_CTX

        backend = OpenAiChatBackend("local", f"{server.process.base_url}/v1/chat/completions", model, None, ref)
        backend.context_window = N_CTX
        return backend


def describe(models: list[AgentModel]) -> list[dict[str, Any]]:
    return [m.model_dump() for m in models]
