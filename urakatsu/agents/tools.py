"""Tools an agent can call, and the context they run in.

A ``Tool`` is a JSON-schema signature plus an async handler. Handlers receive a
``ToolContext`` (the workspace, the project and page open in the UI, and ``emit`` for
live events such as run progress) and return a JSON-serializable result. A handler that
raises is reported to the model as ``{"error": ...}`` so it can correct itself; it never
ends the turn.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .messages import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.service.workspace import Workspace

    from .image_edit import ImageEditor

Handler = Callable[["ToolContext", dict[str, Any]], Any]
# Results longer than this are cut before they go back to the model.
RESULT_LIMIT = 60_000


@dataclass
class ToolContext:
    workspace: "Workspace"
    project_id: Optional[str] = None
    page_id: Optional[str] = None
    emit: Callable[[dict[str, Any]], None] = lambda event: None
    # Set by tools that changed the project or the settings, so the UI reloads them.
    changed: set[str] = field(default_factory=set)
    # The image model ``edit_image`` uses; ``None`` when there is none.
    image_editor: Optional["ImageEditor"] = None

    def project(self) -> str:
        if not self.project_id:
            raise ValueError("no project is open; ask the user to open one")
        return self.project_id

    def page(self, page_id: Optional[str]) -> str:
        page = page_id or self.page_id
        if not page:
            raise ValueError("no page given and none is selected")
        return page


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    # Tools that edit the project or settings; the UI reloads after them.
    mutates: bool = False
    # Whether the model is offered the tool in this context (edit_image needs an image model).
    available: Optional[Callable[["ToolContext"], bool]] = None

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    async def call(self, context: ToolContext, arguments: dict[str, Any]) -> Any:
        result = self.handler(context, arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


def tool(
    name: str,
    description: str,
    properties: Optional[dict[str, Any]] = None,
    required: tuple[str, ...] = (),
    mutates: bool = False,
    available: Optional[Callable[[ToolContext], bool]] = None,
) -> Callable[[Handler], Tool]:
    """Declares a tool from a handler; ``properties`` are the argument JSON schema's properties."""

    def wrap(handler: Handler) -> Tool:
        parameters: dict[str, Any] = {"type": "object", "properties": properties or {}, "additionalProperties": False}
        if required:
            parameters["required"] = list(required)
        return Tool(name, description, parameters, handler, mutates, available)

    return wrap


class Toolset:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self, context: Optional[ToolContext] = None) -> list[ToolSpec]:
        """The tools to offer; with a ``context``, only those available in it."""
        return [t.spec() for t in self._tools.values() if context is None or t.available is None or t.available(context)]

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def call(self, context: ToolContext, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Runs a tool; returns whether it succeeded and the result as JSON text for the model."""
        found = self._tools.get(name)
        if found is None:
            return False, json.dumps({"error": f"unknown tool {name!r}"})
        try:
            result = await found.call(context, arguments)
        except Exception as error:  # reported to the model, which can retry differently
            message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
            return False, json.dumps({"error": str(message) or type(error).__name__}, ensure_ascii=False)
        if found.mutates:
            context.changed.add("project")
        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > RESULT_LIMIT:
            text = text[:RESULT_LIMIT] + "… (truncated)"
        return True, text

