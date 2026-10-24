"""Fitting a conversation into the model's context window.

Every model step sends the whole conversation, and tool results (a page's blocks, a
search) are the bulk of it. ``fit`` returns the messages to send for one step, shortened
to the step's token budget; the saved conversation is never changed, so a later step on
a model with a larger window sees the full results again. In order, until it fits:

1. Tool results from before the newest step are cut to ``STUB_TOKENS``, oldest first.
2. The newest step's results (the ones the model is about to read) share what is left.
3. Whole turns are left out, oldest first; the latest turn always stays.

A shortened result ends with a note that names the tool, so the model can call it again,
narrower, when it needs the rest.

The budget is the backend's ``context_window`` (the local model's ``N_CTX``, a ChatGPT
model's listed window) or ``DEFAULT_CONTEXT_WINDOW``, less a reserve for the reply and the
cost of the instructions and tool schemas. Tokens are estimated, not counted: a third of
the UTF-8 length, which over-counts English and JSON a little and matches CJK text (three
bytes, about one token per character). When a service still says the request was too
long, ``ContextBudget.shrink`` halves the window and the loop sends the step again.

Koharu sends the whole semantic state of every page with each request (koharu-rs/koharu
issue #1098); its unmerged fix (PR #1135) inspects pages on demand and compacts large
inspection results before they persist. Urakatsu's tools already read pages on demand;
this module is the compaction half, budgeted by the model's window instead.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Optional

from .messages import Message, ToolSpec

DEFAULT_CONTEXT_WINDOW = 128_000
# Kept free for the model's reply: this much, or a quarter of a small window.
REPLY_RESERVE = 16_000
# A step is never budgeted below this, whatever the window (the service then decides).
MIN_BUDGET = 1_000
# What an older tool result is cut to.
STUB_TOKENS = 200
# Each message's role and framing.
MESSAGE_OVERHEAD = 4
OMITTED = "[Earlier messages in this conversation were left out to fit the model's context window.]\n\n"


def estimate_tokens(text: str) -> int:
    return -(-len(text.encode("utf-8")) // 3)


def _cost(message: Message) -> int:
    calls = sum(estimate_tokens(call.name) + estimate_tokens(call.arguments_json()) for call in message.tool_calls)
    return MESSAGE_OVERHEAD + estimate_tokens(message.content) + calls


def _cut(text: str, tokens: int) -> str:
    return text.encode("utf-8")[: max(tokens, 0) * 3].decode("utf-8", "ignore")


def shorten(message: Message, tokens: int) -> Message:
    """A tool message cut to about ``tokens``, with a note saying so."""
    full = estimate_tokens(message.content)
    if full <= tokens:
        return message
    note = (
        f"… [result shortened from about {full} tokens to fit the model's context window; "
        f"call {message.name or 'the tool'} again, narrower if possible, for the rest]"
    )
    return dataclasses.replace(message, content=_cut(message.content, tokens - estimate_tokens(note)) + note)


class ContextBudget:
    """The token budget of one turn's model steps."""

    def __init__(self, window: Optional[int]):
        self.window = window or DEFAULT_CONTEXT_WINDOW

    def shrink(self) -> None:
        """The service said the request was too long: assume half the window from now on."""
        self.window = max(self.window // 2, MIN_BUDGET)

    def for_messages(self, system: str, tools: list[ToolSpec]) -> int:
        """Tokens left for the conversation once the reply, instructions and tools are paid for."""
        fixed = estimate_tokens(system) + sum(
            estimate_tokens(t.name) + estimate_tokens(t.description) + estimate_tokens(json.dumps(t.parameters)) for t in tools
        )
        reserve = min(REPLY_RESERVE, self.window // 4)
        return max(self.window - reserve - fixed, MIN_BUDGET)


def fit(messages: list[Message], budget: int) -> list[Message]:
    """The messages to send for one step: ``messages`` itself when they fit ``budget`` tokens,
    otherwise a shortened copy (see the module docstring)."""
    costs = [_cost(m) for m in messages]
    total = sum(costs)
    if total <= budget:
        return messages
    out = list(messages)

    def replace(index: int, message: Message) -> None:
        nonlocal total
        cost = _cost(message)
        total += cost - costs[index]
        out[index], costs[index] = message, cost

    # The newest step's results: everything after the last assistant message.
    newest = next((i + 1 for i in range(len(out) - 1, -1, -1) if out[i].role == "assistant"), 0)
    for index in range(newest):
        if total <= budget:
            return out
        if out[index].role == "tool":
            replace(index, shorten(out[index], STUB_TOKENS))

    fresh = [i for i in range(newest, len(out)) if out[i].role == "tool"]
    if total > budget and fresh:
        others = total - sum(costs[i] for i in fresh)
        share = max((budget - others) // len(fresh) - MESSAGE_OVERHEAD, STUB_TOKENS)
        for index in fresh:
            replace(index, shorten(out[index], share))

    starts = [i for i, m in enumerate(out) if m.role == "user"]
    dropped = 0
    while total > budget and len(starts) > 1:
        cut = starts.pop(1) - dropped
        total -= sum(costs[:cut])
        del out[:cut], costs[:cut]
        dropped += cut
    if dropped:
        out[0] = dataclasses.replace(out[0], content=OMITTED + out[0].content)
    return out
