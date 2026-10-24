"""Slash commands: ``/name text`` in the agent's message box runs one tool directly.

Asked to "translate this page", a model reasonably runs the translation pipeline even when
the user meant an image edit; a command names the tool instead. No model runs: the turn
records the message as typed, then the tool call and its result as if the assistant had
made them, and one line of its own, so the conversation reads the same and a follow-up
message can refer to the result.

A command builds the tool's arguments from the text after its name, the open page, and the
text blocks selected in the editor (``selection``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .tools import ToolContext

Build = Callable[[ToolContext, str, list[str]], dict[str, Any]]


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    # What follows the name, shown as a hint ("" when nothing does).
    argument: str
    tool: str
    build: Build
    # The assistant's line after a successful run.
    done: str

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "argument": self.argument}


def _language(context: ToolContext) -> str:
    tag = context.workspace.config.translation.target_language
    return next((name for code, name in context.workspace.languages() if code == tag), tag)


def _extra(text: str) -> str:
    return f"\n\nAlso: {text}" if text else ""


def _translate_image(context: ToolContext, text: str, selection: list[str]) -> dict[str, Any]:
    return {
        "mode": "page",
        "base": "source",
        "instruction": f"Translate all the lettering on this page into {_language(context)}: speech balloons, captions, "
        "signs and sound effects. Re-letter each one in place, in the same style, size, angle and colour as the "
        "original, with natural phrasing that fits its space." + _extra(text),
    }


def _translate_sfx(context: ToolContext, text: str, selection: list[str]) -> dict[str, Any]:
    if not selection:
        raise ValueError("select the sound effect's text block first (draw one around it with the text tool if detection missed it)")
    return {
        "mode": "region",
        "layer_ids": selection,
        "base": "source",
        "instruction": f"Translate the sound effect in the middle of this image into {_language(context)}. Re-letter it "
        "by hand in the same style, size, angle and colour, and remove the original lettering." + _extra(text),
    }


def _edit_image(context: ToolContext, text: str, selection: list[str]) -> dict[str, Any]:
    if not text:
        raise ValueError("say what to change, e.g. /edit-image remove the speed lines behind her")
    arguments: dict[str, Any] = {"instruction": text, "base": "current", "hide_text": False}
    if selection:
        arguments.update(mode="region", layer_ids=selection)
    else:
        arguments["mode"] = "page"
    return arguments


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in (
        Command(
            "translate-image",
            "Translate the whole page into the art with the image model, hiding its typeset text",
            "[extra instructions]",
            "edit_image",
            _translate_image,
            "Translated the page in the art as a new image edit layer; its typeset text is hidden. One undo reverts both.",
        ),
        Command(
            "translate-sfx",
            "Redraw the selected sound effect in the target language with the image model",
            "[extra instructions]",
            "edit_image",
            _translate_sfx,
            "Redrew the sound effect as a new image edit layer. One undo reverts it.",
        ),
        Command(
            "edit-image",
            "Edit the selected block's area, or the whole page, with the image model",
            "<what to change>",
            "edit_image",
            _edit_image,
            "Added the edit as a new image edit layer. One undo reverts it.",
        ),
    )
}


def parse(text: str) -> Optional[tuple[Command, str]]:
    """The command a message starts with and the text after it; ``None`` when it is not a command."""
    if not text.startswith("/"):
        return None
    head, _, rest = text[1:].partition(" ")
    name, _, more = head.partition("\n")
    command = COMMANDS.get(name.lower())
    if command is None:
        raise ValueError(f"unknown command /{name}; the commands are " + ", ".join(f"/{n}" for n in COMMANDS))
    return command, " ".join(part for part in (more, rest) if part).strip()
