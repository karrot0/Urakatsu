"""Tools an agent can call, and the context they run in.

A ``Tool`` is a JSON-schema signature plus an async handler. Handlers receive a
``ToolContext`` (the workspace, the project and page open in the UI, and ``emit`` for
live events such as run progress) and return a JSON-serializable result. A handler that
raises is reported to the model as ``{"error": ...}`` so it can correct itself; it never
ends the turn.

Every tool has an access level, and a conversation runs at one the user picks
(``ACCESS``): ``read`` tools only look, ``edit`` tools change the open project, and
``full`` tools change the app itself (settings for every stage, model downloads, loaded
models). A tool above the conversation's level is not offered to the model, and calling
it anyway is refused with a message telling the user how to allow it.
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
RESULT_LIMIT = 60_000

ACCESS = ("read", "edit", "full")
DEFAULT_ACCESS = "full"


def access_rank(level: str) -> int:
    try:
        return ACCESS.index(level)
    except ValueError as error:
        raise ValueError(f"unknown access level {level!r}; expected one of {list(ACCESS)}") from error


@dataclass
class ToolContext:
    workspace: "Workspace"
    project_id: Optional[str] = None
    page_id: Optional[str] = None
    emit: Callable[[dict[str, Any]], None] = lambda event: None
    changed: set[str] = field(default_factory=set)
    image_editor: Optional["ImageEditor"] = None
    access: str = DEFAULT_ACCESS

    def allows(self, level: str) -> bool:
        return access_rank(level) <= access_rank(self.access)

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
    mutates: bool = False
    available: Optional[Callable[["ToolContext"], bool]] = None
    access: str = ""

    def __post_init__(self) -> None:
        self.access = self.access or ("edit" if self.mutates else "read")
        access_rank(self.access)

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
    access: str = "",
) -> Callable[[Handler], Tool]:
    """Declares a tool from a handler; ``properties`` are the argument JSON schema's properties."""

    def wrap(handler: Handler) -> Tool:
        parameters: dict[str, Any] = {"type": "object", "properties": properties or {}, "additionalProperties": False}
        if required:
            parameters["required"] = list(required)
        return Tool(name, description, parameters, handler, mutates, available, access)

    return wrap


class Toolset:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self, context: Optional[ToolContext] = None) -> list[ToolSpec]:
        """The tools to offer; with a ``context``, only those available in it and within its access."""
        return [t.spec() for t in self._tools.values() if context is None or self._offered(t, context)]

    @staticmethod
    def _offered(found: Tool, context: ToolContext) -> bool:
        return context.allows(found.access) and (found.available is None or found.available(context))

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def call(self, context: ToolContext, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Runs a tool; returns whether it succeeded and the result as JSON text for the model."""
        found = self._tools.get(name)
        if found is None:
            return False, json.dumps({"error": f"unknown tool {name!r}"})
        if not context.allows(found.access):
            needed = "Full access" if found.access == "full" else "Edit project"
            return False, json.dumps(
                {"error": f"{name} needs {found.access} access and this chat has {context.access}; ask the user to switch the chat's access to {needed}"}
            )
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
