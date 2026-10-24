"""Provider-neutral conversation records and the events a chat backend streams.

A conversation is a list of ``Message``: user text, assistant text with the tool calls it
asked for, and one tool message per call carrying the tool's JSON result. Each backend
translates this list to its own wire format, so a conversation can switch models
between turns.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Tool messages: the call they answer.
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # Assistant messages: the model that wrote them (``provider:model``).
    model: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Message":
        calls = [ToolCall(**call) for call in data.get("tool_calls") or []]
        return cls(**{**data, "tool_calls": calls})


@dataclass
class ToolSpec:
    """A tool as the model sees it: a name, what it does, and a JSON schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


# -- backend stream events -------------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str  # JSON text, as the model wrote it


BackendEvent = Union[TextDelta, ReasoningDelta, ToolCallRequest]


def parse_arguments(text: str) -> dict[str, Any]:
    """A tool call's arguments; an empty string is no arguments, and malformed JSON is an error."""
    if not text.strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value
