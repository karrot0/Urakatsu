"""Agents: a persona (instructions) plus the tools it may use.

An agent knows nothing about which model runs it; the conversation picks the model
(``models.py``) and the loop (``session.py``) wires the two together. New agents
subclass ``Agent`` and register with ``@register_agent``.
"""

from __future__ import annotations

from typing import ClassVar

from .app_tools import APP_TOOLS
from .tools import ToolContext, Toolset

_agents: dict[str, type["Agent"]] = {}


def register_agent(cls: type["Agent"]) -> type["Agent"]:
    _agents[cls.id] = cls
    return cls


def agent_classes() -> list[type["Agent"]]:
    return list(_agents.values())


def get_agent(agent_id: str) -> "Agent":
    try:
        return _agents[agent_id]()
    except KeyError as error:
        raise KeyError(f"unknown agent {agent_id!r}") from error


class Agent:
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]

    def toolset(self) -> Toolset:
        return Toolset([])

    def instructions(self, context: ToolContext) -> str:
        raise NotImplementedError  # pragma: no cover - interface


@register_agent
class Assistant(Agent):
    """The editor's assistant: reads and edits the open project and drives the pipeline."""

    id: ClassVar[str] = "assistant"
    name: ClassVar[str] = "Assistant"
    description: ClassVar[str] = "Reviews and edits translations, typesetting and runs in the open project"

    def toolset(self) -> Toolset:
        return Toolset(APP_TOOLS)

    def instructions(self, context: ToolContext) -> str:
        workspace = context.workspace
        lines = [
            "You are the assistant inside Urakatsu, a desktop app that translates manga, manhwa and comics.",
            "A project holds chapters of pages. Each page goes through four pipeline stages: detection (finds text "
            "regions and balloons), ocr (reads the source text), translation, and inpainting (removes the original "
            "lettering); the app then typesets the translations into the page.",
            "You work in the user's open project through tools. Read before you write: call get_project or get_page "
            "to see ids and the current text, never guess ids. Your edits count as the user's, so pipeline re-runs "
            "keep them, and every edit can be undone.",
            "When asked to improve translations, keep names, honorifics and terms consistent across pages "
            "(search_text helps), keep lines short enough for their balloons, and preserve the speaker's tone.",
            "Sound effects and other stylized lettering are redrawn in the art with edit_image, not typeset. You cannot "
            "see the page, so take the region from text block frames (get_page) or ask the user where it is.",
            "Before long or broad operations (running the whole project, deleting blocks, changing settings), say "
            "what you are about to do. Keep replies brief; the user sees your tool calls.",
        ]
        if context.project_id:
            try:
                view = workspace.project_view(context.project_id)
                lines.append(f"Open project: {view.name} (id {view.id}, {len(view.pages)} pages, {len(view.chapters)} chapters).")
                page = next((p for p in view.pages if p.id == context.page_id), None)
                if page is not None:
                    lines.append(f"Open page: {page.name} (id {page.id}).")
            except KeyError:
                pass
        else:
            lines.append("No project is open; most tools need one.")
        translation = workspace.config.translation
        lines.append(
            f"Translation settings: provider {translation.model.provider}, model {translation.model.model or 'default'}, "
            f"target language {translation.target_language}."
        )
        return "\n\n".join(lines)
