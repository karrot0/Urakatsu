"""Conversations and the agent loop.

A conversation is saved as JSON under ``~/.urakatsu/agents/conversations``. Sending a
message starts a turn in the background: the loop asks the model for a step, runs the
tools it calls, feeds their results back, and repeats until the model answers without
calling a tool (or ``MAX_STEPS`` is reached). Turn events are buffered like pipeline job
events, so the UI can leave the panel and re-attach to a running turn.

Each step sends the conversation fitted to the model's context window (``compaction.py``).
A step that fails in a way that may pass (``backends.base.failure_kind``) is sent again:
a transient failure after each of ``RETRY_DELAYS``, a context overflow up to
``OVERFLOW_RETRIES`` times with the budget halved each time. A retry publishes a
``retry`` event, and the UI drops the text the failed attempt had streamed. Tools run only
once a step has streamed to the end, so a retried step has changed nothing.

A slash command (``commands.py``) runs its tool without a model (``run_command``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from urakatsu.core.config import home

from .agent import Agent, get_agent
from .backends.base import ChatBackend, failure_kind
from .commands import Command
from .compaction import ContextBudget, fit
from .messages import Message, ReasoningDelta, TextDelta, ToolCall, ToolCallRequest, ToolSpec, new_id, parse_arguments
from .tools import ToolContext

log = logging.getLogger(__name__)

MAX_STEPS = 40
# Seconds to wait before each new attempt at a step that failed transiently.
RETRY_DELAYS = (1.0, 3.0, 8.0)
OVERFLOW_RETRIES = 2
# Tool results shown in the UI are cut to this; the model gets up to ``tools.RESULT_LIMIT``.
UI_RESULT_LIMIT = 4000


@dataclass
class Conversation:
    id: str
    agent: str
    model: str
    project_id: Optional[str] = None
    title: str = "New chat"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[Message] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "model": self.model,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> dict[str, Any]:
        return {**self.summary(), "messages": [m.to_json() for m in self.messages]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Conversation":
        messages = [Message.from_json(m) for m in data.get("messages") or []]
        return cls(**{**data, "messages": messages})


class ConversationStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or home() / "agents" / "conversations"

    def _path(self, conversation_id: str) -> Path:
        if not conversation_id.replace("_", "").isalnum():
            raise KeyError(f"no conversation {conversation_id}")
        return self.root / f"{conversation_id}.json"

    def create(self, agent: str, model: str, project_id: Optional[str]) -> Conversation:
        get_agent(agent)
        conversation = Conversation(id=new_id("chat"), agent=agent, model=model, project_id=project_id)
        self.save(conversation)
        return conversation

    def get(self, conversation_id: str) -> Conversation:
        path = self._path(conversation_id)
        if not path.exists():
            raise KeyError(f"no conversation {conversation_id}")
        return Conversation.from_json(json.loads(path.read_text(encoding="utf-8")))

    def save(self, conversation: Conversation) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        conversation.updated_at = time.time()
        path = self._path(conversation.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(conversation.to_json(), ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def delete(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)

    def list(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        found = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if project_id is None or data.get("project_id") == project_id:
                data.pop("messages", None)
                found.append(data)
        return sorted(found, key=lambda d: d.get("updated_at", 0), reverse=True)


@dataclass
class Turn:
    conversation_id: str
    status: str = "running"  # running | done | stopped | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    queues: list[asyncio.Queue] = field(default_factory=list)
    task: Optional[asyncio.Task] = None

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for queue in list(self.queues):
            queue.put_nowait(event)

    def finish(self) -> None:
        for queue in list(self.queues):
            queue.put_nowait(None)


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= 60 else line[:59].rstrip() + "…"


async def run_turn(
    conversation: Conversation,
    agent: Agent,
    backend: ChatBackend,
    context: ToolContext,
    publish: Callable[[dict[str, Any]], None],
    save: Callable[[Conversation], None],
) -> str:
    """Runs model steps and tool calls until the model stops calling tools; returns the final status."""
    toolset = agent.toolset()
    specs = toolset.specs(context)
    budget = ContextBudget(backend.context_window)
    context.emit = publish
    text: list[str] = []
    try:
        for _ in range(MAX_STEPS):
            text = []
            publish({"type": "step", "model": backend.ref})
            requests = await _model_step(conversation, agent, backend, context, specs, budget, publish, text)
            calls: list[tuple[ToolCall, Optional[str]]] = []
            for request in requests:
                try:
                    calls.append((ToolCall(request.id, request.name, parse_arguments(request.arguments)), None))
                except ValueError as error:
                    calls.append((ToolCall(request.id, request.name, {}), f"arguments were not valid JSON: {error}"))
            conversation.messages.append(Message("assistant", "".join(text), [call for call, _ in calls], model=backend.ref))
            text = []
            save(conversation)
            if not calls:
                return "done"
            for call, problem in calls:
                publish({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments})
                if problem is None:
                    ok, result = await toolset.call(context, call.name, call.arguments)
                else:
                    ok, result = False, json.dumps({"error": problem})
                conversation.messages.append(Message("tool", result, tool_call_id=call.id, name=call.name))
                save(conversation)
                publish(
                    {
                        "type": "tool_result",
                        "id": call.id,
                        "name": call.name,
                        "ok": ok,
                        "result": result[:UI_RESULT_LIMIT],
                        "changed": sorted(context.changed),
                    }
                )
                context.changed.clear()
        publish({"type": "error", "message": f"Stopped after {MAX_STEPS} steps without a final answer."})
        return "failed"
    except asyncio.CancelledError:
        _close_open_calls(conversation, text, backend.ref, "stopped by the user")
        save(conversation)
        return "stopped"
    except Exception as error:
        log.warning("agent turn failed: %s", error, exc_info=not isinstance(error, (ValueError, KeyError)))
        _close_open_calls(conversation, text, backend.ref, "the turn failed")
        save(conversation)
        publish({"type": "error", "message": str(error) or type(error).__name__})
        return "failed"


async def _model_step(
    conversation: Conversation,
    agent: Agent,
    backend: ChatBackend,
    context: ToolContext,
    specs: list[ToolSpec],
    budget: ContextBudget,
    publish: Callable[[dict[str, Any]], None],
    text: list[str],
) -> list[ToolCallRequest]:
    """Streams one model step into ``text`` and returns the tool calls it asked for, sending
    the step again after a failure that may pass."""
    retries = overflows = 0
    while True:
        requests: list[ToolCallRequest] = []
        system = agent.instructions(context)
        messages = fit(conversation.messages, budget.for_messages(system, specs))
        try:
            async for event in backend.stream(system, messages, specs):
                if isinstance(event, TextDelta):
                    text.append(event.text)
                    publish({"type": "text", "delta": event.text})
                elif isinstance(event, ReasoningDelta):
                    publish({"type": "reasoning", "delta": event.text})
                else:
                    requests.append(event)
            return requests
        except Exception as error:
            kind = failure_kind(error)
            if kind == "overflow" and overflows < OVERFLOW_RETRIES:
                overflows += 1
                budget.shrink()
                delay = 0.0
            elif kind == "transient" and retries < len(RETRY_DELAYS):
                delay = RETRY_DELAYS[retries]
                retries += 1
            else:
                raise
            log.info("agent step failed (%s), sending it again in %.0fs: %s", kind, delay, error)
            publish({"type": "retry", "reason": kind, "delay": delay, "message": str(error) or type(error).__name__})
            text.clear()  # the UI dropped it too; a stop during the wait keeps none of it
            await asyncio.sleep(delay)


async def run_command(
    conversation: Conversation,
    agent: Agent,
    command: Command,
    arguments: dict[str, Any],
    context: ToolContext,
    publish: Callable[[dict[str, Any]], None],
    save: Callable[[Conversation], None],
) -> str:
    """Runs a slash command's tool as the assistant's one call, then says how it went."""
    context.emit = publish
    call = ToolCall(new_id("call"), command.tool, arguments)
    conversation.messages.append(Message("assistant", "", [call]))
    save(conversation)
    publish({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments})
    try:
        ok, result = await agent.toolset().call(context, call.name, call.arguments)
    except asyncio.CancelledError:
        _close_open_calls(conversation, [], "", "stopped by the user")
        save(conversation)
        return "stopped"
    conversation.messages.append(Message("tool", result, tool_call_id=call.id, name=call.name))
    publish({"type": "tool_result", "id": call.id, "name": call.name, "ok": ok, "result": result[:UI_RESULT_LIMIT], "changed": sorted(context.changed)})
    context.changed.clear()
    if ok:
        text = command.done
    else:
        try:
            reason = json.loads(result).get("error") or result
        except (ValueError, AttributeError):
            reason = result
        text = f"/{command.name} failed: {reason}"
    conversation.messages.append(Message("assistant", text))
    save(conversation)
    publish({"type": "text", "delta": text})
    return "done" if ok else "failed"


def _close_open_calls(conversation: Conversation, text: list[str], model: str, reason: str) -> None:
    """Keeps the transcript valid after an interruption: partial text is kept and every
    tool call gets a result, since providers reject a call without one."""
    if text:
        conversation.messages.append(Message("assistant", "".join(text), model=model or None))
    answered = {m.tool_call_id for m in conversation.messages if m.role == "tool"}
    for message in reversed(conversation.messages):
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.id not in answered:
                    conversation.messages.append(Message("tool", json.dumps({"error": reason}), tool_call_id=call.id, name=call.name))
            break


class Turns:
    """The running and last turn of each conversation."""

    def __init__(self, store: ConversationStore):
        self.store = store
        self.turns: dict[str, Turn] = {}

    def running(self, conversation_id: str) -> bool:
        turn = self.turns.get(conversation_id)
        return turn is not None and turn.status == "running"

    def start(
        self,
        conversation: Conversation,
        text: str,
        agent: Agent,
        backend: Optional[ChatBackend],
        context: ToolContext,
        command: Optional[tuple[Command, dict[str, Any]]] = None,
    ) -> Turn:
        """Starts a model turn, or with ``command`` (and its tool arguments) runs a slash command, which needs no backend."""
        if self.running(conversation.id):
            raise ValueError("the assistant is still answering; stop it first")
        if not conversation.messages:
            conversation.title = _title(text)
        if backend is not None:
            conversation.model = backend.ref
        conversation.messages.append(Message("user", text))
        self.store.save(conversation)
        turn = Turn(conversation.id)
        self.turns[conversation.id] = turn
        turn.publish({"type": "user", "content": text})

        async def runner() -> None:
            try:
                if command is not None:
                    turn.status = await run_command(conversation, agent, command[0], command[1], context, turn.publish, self.store.save)
                else:
                    assert backend is not None
                    turn.status = await run_turn(conversation, agent, backend, context, turn.publish, self.store.save)
            finally:
                turn.publish({"type": "done", "status": turn.status})
                turn.finish()

        turn.task = asyncio.create_task(runner())
        return turn

    def stop(self, conversation_id: str) -> None:
        turn = self.turns.get(conversation_id)
        if turn is not None and turn.task is not None and not turn.task.done():
            turn.task.cancel()

    async def events(self, conversation_id: str) -> AsyncIterator[dict[str, Any]]:
        turn = self.turns.get(conversation_id)
        if turn is None:
            return
        queue: asyncio.Queue = asyncio.Queue()
        for event in turn.events:
            queue.put_nowait(event)
        if turn.status != "running":
            queue.put_nowait(None)
        else:
            turn.queues.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            if queue in turn.queues:
                turn.queues.remove(queue)

    async def close(self) -> None:
        tasks = [t.task for t in self.turns.values() if t.task is not None and not t.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
