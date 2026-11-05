"""Agents: a persona (instructions) plus the tools it may use.

An agent knows nothing about which model runs it; the conversation picks the model
(``models.py``) and the loop (``session.py``) wires the two together. New agents
subclass ``Agent`` and register with ``@register_agent``.
"""

from __future__ import annotations

from typing import ClassVar

from .app_tools import APP_TOOLS, fill_model
from .stage_tools import STAGE_TOOLS
from .tools import ToolContext, Toolset

_agents: dict[str, type["Agent"]] = {}

_ACCESS_NOTES = {
    "read": "The user gave you read-only access in this chat: you can look at the project, the pipeline and the "
    "models, but not change anything. When a request needs a change, say what you would do and that they can "
    "allow it with the access menu under the message box.",
    "edit": "The user gave you edit access in this chat: you can edit the open project and run the pipeline, but not "
    "change settings, stages or models. When a request needs that, say so and that they can allow Full access with "
    "the access menu under the message box.",
    "full": "The user gave you full access in this chat: you can edit the project, run the pipeline, change every "
    "stage's provider, model and settings, and install models.",
}


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


def _fill_note(context: ToolContext) -> str:
    model = fill_model(context)
    if not model["prompted"]:
        how = (
            "it only removes what is in the area (it takes no prompt); with full access, configure_stage on 'fill' "
            "can switch it to a diffusion model that follows prompts"
        )
    elif model["instructed"]:
        how = "it follows edit instructions, so phrase the prompt as what to change"
    elif model["instructed"] is None:
        how = "whether it takes edit instructions or descriptions depends on the installed model; say plainly what the area should show"
    else:
        how = "it takes a description of what belongs in the area"
    return (
        f"fill_region repaints part of a page with the local generative fill model, {model['name']}: {how}. Use it for "
        "local edits, removals and redrawn lettering on this machine; edit_image uses a hosted image model instead."
    )


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
        return Toolset([*APP_TOOLS, *STAGE_TOOLS])

    def instructions(self, context: ToolContext) -> str:
        workspace = context.workspace
        lines = [
            "You are the assistant inside Urakatsu, a desktop app that translates manga, manhwa and comics.",
            "A project holds chapters of pages. Each page goes through pipeline stages: detection (finds text "
            "regions and balloons), ocr (reads the source text), translation, and inpainting (removes the original "
            "lettering), plus optional ones (watermarks, censor_detection, decensor, colorization, sound_effects) "
            "that run only when their mode is on; the app then typesets the translations into the page.",
            "Every stage is run by a swappable provider and model. get_pipeline shows what runs each stage; "
            "list_stage_providers shows the alternatives and the installed models each can use; configure_stage "
            "switches a stage's provider, runtime, model, mode or settings; search_models, get_model_details and "
            "install_model get new models from the Hugging Face catalog. To try a different model on a page, change "
            "the stage, then run_pipeline with mode only on that stage.",
            "You work in the user's open project through tools. Read before you write: call get_project or get_page "
            "to see ids and the current text, never guess ids. Your edits count as the user's, so pipeline re-runs "
            "keep them, and every edit can be undone.",
            "When asked to improve translations, keep names, honorifics and terms consistent across pages "
            "(search_text helps), keep lines short enough for their balloons, and preserve the speaker's tone.",
            "The project's knowledge base (get_knowledge, save_knowledge) is what the translator sees of the series: "
            "asked to build a glossary, read the chapters' text and save names, terms and facts there.",
            (
                "Sound effects and other stylized lettering are redrawn in the art with edit_image, not typeset. You "
                "cannot see the page, so take the region from text block frames (get_page) or ask the user where it is."
                if context.image_editor is not None
                else "Sound effects and other stylized lettering need an image model, and none is available: tell the "
                "user to sign in with ChatGPT, or to allow image generation on their Urakatsu Cloud key."
            ),
            _fill_note(context),
            "Before long or broad operations (running the whole project, deleting blocks, changing settings, "
            "downloading a model), say what you are about to do. Keep replies brief; the user sees your tool calls.",
            _ACCESS_NOTES[context.access],
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
