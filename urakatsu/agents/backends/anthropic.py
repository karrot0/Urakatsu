"""Anthropic Messages with tools, streamed (Claude)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from urakatsu.providers.translation.remote.claude import API_VERSION, URL

from ..messages import BackendEvent, Message, ReasoningDelta, TextDelta, ToolCallRequest, ToolSpec
from .base import AgentError, ChatBackend, overflowed, sse_events

MAX_TOKENS = 16_000
# Error event types (sent mid-stream, after a 200) that may pass if the step is sent again.
TRANSIENT_ERRORS = ("overloaded_error", "api_error", "rate_limit_error", "timeout_error")


def wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Tool results ride in a user turn; consecutive results share one."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            wire.append({"role": "user", "content": [{"type": "text", "text": message.content}]})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in message.tool_calls]
            wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
        else:
            result = {"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}
            if wire and wire[-1]["role"] == "user" and wire[-1]["content"] and wire[-1]["content"][-1].get("type") == "tool_result":
                wire[-1]["content"].append(result)
            else:
                wire.append({"role": "user", "content": [result]})
    return wire


class AnthropicBackend(ChatBackend):
    def __init__(self, model: str, api_key: str, ref: str = ""):
        self.model = model
        self.api_key = api_key
        self.ref = ref or f"claude:{model}"

    async def stream(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AsyncIterator[BackendEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": wire_messages(messages),
            "stream": True,
        }
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION}
        blocks: dict[int, dict[str, str]] = {}
        async for event, data in sse_events("claude", URL, headers=headers, body=body, transport=self.transport):
            kind = data.get("type", event)
            if kind == "content_block_start":
                block = data.get("content_block") or {}
                if block.get("type") == "tool_use":
                    blocks[int(data["index"])] = {"id": block["id"], "name": block["name"], "arguments": ""}
            elif kind == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield TextDelta(delta.get("text", ""))
                elif delta.get("type") == "thinking_delta":
                    yield ReasoningDelta(delta.get("thinking", ""))
                elif delta.get("type") == "input_json_delta" and int(data["index"]) in blocks:
                    blocks[int(data["index"])]["arguments"] += delta.get("partial_json", "")
            elif kind == "error":
                error = data.get("error") or {}
                message = error.get("message") or json.dumps(data)[:300]
                raise AgentError(
                    f"claude: {message}",
                    transient=error.get("type") in TRANSIENT_ERRORS,
                    overflow=error.get("type") == "invalid_request_error" and overflowed(message),
                )
        for index in sorted(blocks):
            block = blocks[index]
            yield ToolCallRequest(block["id"], block["name"], block["arguments"] or "{}")

