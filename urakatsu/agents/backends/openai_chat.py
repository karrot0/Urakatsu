"""OpenAI Chat Completions with function tools, streamed.

Most hosted LLM providers and llama.cpp's ``llama-server`` speak this format, so it
backs OpenAI, OpenRouter, DeepSeek, xAI, MiniMax, Gemini (its OpenAI endpoint), LM
Studio, any OpenAI-compatible server, and the loaded local model.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from ..messages import BackendEvent, Message, ReasoningDelta, TextDelta, ToolCallRequest, ToolSpec
from .base import ChatBackend, sse_events


def wire_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "user":
            wire.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments_json()}}
                    for call in message.tool_calls
                ]
            wire.append(entry)
        else:
            wire.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content})
    return wire


def wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]


class OpenAiChatBackend(ChatBackend):
    def __init__(self, provider: str, url: str, model: str, api_key: Optional[str] = None, ref: str = "", extra: Optional[dict[str, Any]] = None):
        self.provider = provider
        self.url = url
        self.model = model
        self.api_key = api_key
        self.ref = ref or f"{provider}:{model}"
        self.extra = extra or {}

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def stream(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[BackendEvent]:
        body: dict[str, Any] = {"model": self.model, "messages": wire_messages(system, messages), "stream": True, **self.extra}
        if tools:
            body["tools"] = wire_tools(tools)
        calls: dict[int, dict[str, str]] = {}
        async for _, chunk in sse_events(self.provider, self.url, headers=self.headers(), body=body, transport=self.transport):
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    yield ReasoningDelta(reasoning)
                if delta.get("content"):
                    yield TextDelta(delta["content"])
                for part in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(part.get("index", len(calls))), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = part.get("id") or slot["id"]
                    function = part.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
        for index in sorted(calls):
            slot = calls[index]
            yield ToolCallRequest(slot["id"] or f"call_{index}", slot["name"], slot["arguments"])
