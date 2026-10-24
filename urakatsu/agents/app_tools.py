"""Urakatsu's own tools: read and edit the open project, run the pipeline, adjust translation.

Every edit goes through the workspace like a UI edit does, so it is committed with user
authorship (a later pipeline run never overwrites it) and is one undo step.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

from urakatsu.core.stage import Operation, Stage

from .image_edit import QUALITIES, Box, edit_region
from .tools import Tool, ToolContext, tool

STAGES = [s.value for s in Stage.all()]
_PAGE_ID = {"type": "string", "description": "Page id; defaults to the page open in the editor."}
_LAYER_IDS = {"type": "array", "items": {"type": "string"}, "description": "Text layer ids (``layer_id`` from get_page)."}


def _box(bounds) -> Optional[dict[str, int]]:
    if bounds is None:
        return None
    return {"x": round(bounds.x), "y": round(bounds.y), "width": round(bounds.width), "height": round(bounds.height)}


def _block(block) -> dict[str, Any]:
    out: dict[str, Any] = {"layer_id": block.layer_id, "role": block.role, "source": block.source_text, "translation": block.translation}
    if block.source_origin == "user" or block.translation_origin == "user":
        out["edited_by_user"] = True
    if block.bounds is not None:
        out["bounds"] = _box(block.bounds)
    if not block.visible:
        out["hidden"] = True
    if block.typography is not None:
        typography = block.typography.model_dump(exclude_none=True)
        if typography != {"auto_fit": True}:
            out["typography"] = typography
    return out


def _stages(page) -> dict[str, bool]:
    return {status.stage: status.done for status in page.stages}


def _chapter_pages(context: ToolContext, chapter_id: Optional[str]) -> list[str]:
    view = context.workspace.project_view(context.project())
    if chapter_id:
        for chapter in view.chapters:
            if chapter.id == chapter_id:
                return chapter.page_ids
        raise KeyError(f"no chapter {chapter_id}")
    page = context.page(None)
    for chapter in view.chapters:
        if page in chapter.page_ids:
            return chapter.page_ids
    raise ValueError("the open page is in no chapter; pass chapter_id")


# -- reading --------------------------------------------------------------------------------------


@tool("list_projects", "List every project in the library with its page and chapter counts.")
def list_projects(context: ToolContext, args: dict[str, Any]) -> Any:
    return [
        {"id": p.id, "name": p.name, "pages": p.pages, "chapters": p.chapters, "open": p.id == context.project_id}
        for p in context.workspace.list_projects()
    ]


@tool(
    "get_project",
    "The open project: its chapters, and for each page its id, name, how many text blocks it has, how many are "
    "translated, and which pipeline stages (detection, ocr, translation, inpainting) have run.",
)
def get_project(context: ToolContext, args: dict[str, Any]) -> Any:
    view = context.workspace.project_view(context.project())
    pages = {p.id: p for p in view.pages}
    chapters = []
    for chapter in view.chapters:
        rows = []
        for page_id in chapter.page_ids:
            page = pages.get(page_id)
            if page is None:
                continue
            rows.append(
                {
                    "id": page.id,
                    "name": page.name,
                    "blocks": len(page.blocks),
                    "translated": sum(1 for b in page.blocks if (b.translation or "").strip()),
                    "stages": _stages(page),
                    "open": page.id == context.page_id,
                }
            )
        chapters.append({"id": chapter.id, "name": chapter.name, "pages": rows})
    return {"id": view.id, "name": view.name, "chapters": chapters, "can_undo": view.can_undo, "can_redo": view.can_redo}


@tool(
    "get_page",
    "One page's text blocks in reading order: layer_id, role, source text (OCR), translation, bounds in page "
    "pixels, typography when set, and whether the user edited it. Also the page size and stage status.",
    {"page_id": _PAGE_ID},
)
def get_page(context: ToolContext, args: dict[str, Any]) -> Any:
    page = context.workspace.page(context.project(), context.page(args.get("page_id")))
    return {
        "id": page.id,
        "name": page.name,
        "width": page.width,
        "height": page.height,
        "stages": _stages(page),
        "blocks": [_block(b) for b in page.blocks],
        "raster_layers": [{"id": r.id, "name": r.name, "kind": r.kind, "visible": r.visible} for r in page.raster_layers],
    }


@tool(
    "search_text",
    "Find text blocks across the open project whose source text or translation contains a phrase "
    "(case-insensitive). Useful to keep names and terms consistent.",
    {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
    required=("query",),
)
def search_text(context: ToolContext, args: dict[str, Any]) -> Any:
    needle = str(args["query"]).lower()
    limit = int(args.get("limit") or 50)
    view = context.workspace.project_view(context.project())
    hits = []
    for page in view.pages:
        for block in page.blocks:
            if needle in (block.source_text or "").lower() or needle in (block.translation or "").lower():
                hits.append({"page_id": page.id, "page": page.name, "layer_id": block.layer_id, "source": block.source_text, "translation": block.translation})
                if len(hits) >= limit:
                    return {"hits": hits, "truncated": True}
    return {"hits": hits, "truncated": False}


@tool("get_translation_settings", "The translation provider and model, target language, and custom instructions.")
def get_translation_settings(context: ToolContext, args: dict[str, Any]) -> Any:
    translation = context.workspace.config.translation
    return {
        "provider": translation.model.provider,
        "model": translation.model.model,
        "target_language": translation.target_language,
        "instructions": translation.instructions,
        "languages": [tag for tag, _ in context.workspace.languages()],
    }


@tool("list_translation_models", "Every translation model the configured providers offer (provider id and model id).")
async def list_translation_models(context: ToolContext, args: dict[str, Any]) -> Any:
    models = await asyncio.to_thread(context.workspace.translation_models)
    return [{"provider": m["provider"], "model": m["model"], "name": m["name"], "vision": m.get("vision", False)} for m in models]


@tool(
    "list_fonts",
    "Installed font families for typesetting, optionally filtered by a substring.",
    {"query": {"type": "string"}},
)
async def list_fonts(context: ToolContext, args: dict[str, Any]) -> Any:
    families = [name for name, _ in await asyncio.to_thread(context.workspace.fonts)]
    query = (args.get("query") or "").lower()
    matches = sorted(f for f in families if query in f.lower())
    return {"families": matches[:200], "total": len(matches)}


# -- editing --------------------------------------------------------------------------------------


@tool(
    "edit_text_blocks",
    "Set the translation and/or source text of text blocks on one page. Edits are recorded as the user's, "
    "so later pipeline runs keep them. Each edit is one undo step.",
    {
        "page_id": _PAGE_ID,
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": "string"},
                    "translation": {"type": "string"},
                    "source_text": {"type": "string"},
                },
                "required": ["layer_id"],
                "additionalProperties": False,
            },
        },
    },
    required=("edits",),
    mutates=True,
)
def edit_text_blocks(context: ToolContext, args: dict[str, Any]) -> Any:
    project, page = context.project(), context.page(args.get("page_id"))
    done = []
    for edit in args["edits"]:
        patch = {key: edit[key] for key in ("translation", "source_text") if key in edit}
        if patch:
            context.workspace.update_text_block(project, page, edit["layer_id"], patch)
            done.append(edit["layer_id"])
    return {"updated": done}


_TYPOGRAPHY = {
    "font": {"type": "string", "description": "Font family name from list_fonts; empty for the default."},
    "size": {"type": "number", "description": "Fixed size in points; omit and set auto_fit to fit the frame."},
    "auto_fit": {"type": "boolean"},
    "weight": {"type": "integer", "minimum": 100, "maximum": 900},
    "style": {"type": "string", "enum": ["normal", "italic", "oblique"]},
    "color": {"type": "string", "description": "#rrggbb or #rrggbbaa"},
    "stroke_color": {"type": "string", "description": "#rrggbb or #rrggbbaa; alpha 00 removes the border"},
    "stroke_width": {"type": "number"},
    "alignment": {"type": "string", "enum": ["Start", "Center", "End"]},
    "writing_mode": {"type": "string", "enum": ["", "Horizontal", "Vertical"], "description": "Empty for automatic."},
}
_TYPOGRAPHY_KEYS = {
    "font": "preferred_font",
    "size": "size",
    "auto_fit": "auto_fit",
    "weight": "font_weight",
    "style": "font_style",
    "color": "color",
    "stroke_color": "stroke_color",
    "stroke_width": "stroke_width",
    "alignment": "alignment",
    "writing_mode": "writing_mode",
}


def _hex(value: str) -> str:
    value = value.strip()
    return value + "ff" if len(value) == 7 else value


@tool(
    "set_typography",
    "Change how text blocks are typeset: font, size (or auto fit), weight, style, colour, border and alignment.",
    {"page_id": _PAGE_ID, "layer_ids": _LAYER_IDS, **_TYPOGRAPHY},
    required=("layer_ids",),
    mutates=True,
)
def set_typography(context: ToolContext, args: dict[str, Any]) -> Any:
    project, page = context.project(), context.page(args.get("page_id"))
    patch: dict[str, Any] = {}
    for key, target in _TYPOGRAPHY_KEYS.items():
        if key in args:
            value = args[key]
            patch[target] = _hex(value) if key in ("color", "stroke_color") and value else value
    if "size" in patch and "auto_fit" not in patch:
        patch["auto_fit"] = False
    for layer_id in args["layer_ids"]:
        context.workspace.update_text_block(project, page, layer_id, {"typography": patch})
    return {"updated": args["layer_ids"], "typography": patch}


@tool(
    "set_layer_visibility",
    "Show or hide layers, or set their opacity (0 to 1).",
    {"page_id": _PAGE_ID, "layer_ids": _LAYER_IDS, "visible": {"type": "boolean"}, "opacity": {"type": "number", "minimum": 0, "maximum": 1}},
    required=("layer_ids",),
    mutates=True,
)
def set_layer_visibility(context: ToolContext, args: dict[str, Any]) -> Any:
    page = context.workspace.set_layer_visibility(context.project(), context.page(args.get("page_id")), args["layer_ids"], args.get("visible"), args.get("opacity"))
    return {"updated": args["layer_ids"], "revision": page.revision}


@tool(
    "create_text_block",
    "Add a text block with its own rectangular frame (page pixels), optionally with a translation.",
    {
        "page_id": _PAGE_ID,
        "x": {"type": "number"},
        "y": {"type": "number"},
        "width": {"type": "number"},
        "height": {"type": "number"},
        "translation": {"type": "string"},
    },
    required=("x", "y", "width", "height"),
    mutates=True,
)
def create_text_block(context: ToolContext, args: dict[str, Any]) -> Any:
    project, page_id = context.project(), context.page(args.get("page_id"))
    bounds = {key: float(args[key]) for key in ("x", "y", "width", "height")}
    page = context.workspace.create_text_block(project, page_id, bounds)
    layer_id = page.blocks[-1].layer_id
    if args.get("translation"):
        context.workspace.update_text_block(project, page_id, layer_id, {"translation": args["translation"]})
    return {"layer_id": layer_id}


@tool(
    "delete_text_blocks",
    "Delete text blocks from a page. This can be undone.",
    {"page_id": _PAGE_ID, "layer_ids": _LAYER_IDS},
    required=("layer_ids",),
    mutates=True,
)
def delete_text_blocks(context: ToolContext, args: dict[str, Any]) -> Any:
    project, page = context.project(), context.page(args.get("page_id"))
    for layer_id in args["layer_ids"]:
        context.workspace.delete_text_block(project, page, layer_id)
    return {"deleted": args["layer_ids"]}


_EDIT_PROMPT = (
    "{scope}\n\n{instruction}\n\n"
    "Change only what the instruction asks for{focus}. Keep everything else exactly as it is: the artwork, line "
    "weight, screentones, colours, framing and composition. Keep the whole image the same size and do not add borders."
)


def _region_box(args: dict[str, Any], blocks: dict[str, Any]) -> tuple[int, int, int, int]:
    """The union of ``region`` and the frames of ``layer_ids``, in page pixels."""
    boxes = []
    if args.get("region"):
        r = args["region"]
        boxes.append((float(r["x"]), float(r["y"]), float(r["x"]) + float(r["width"]), float(r["y"]) + float(r["height"])))
    for layer_id in args.get("layer_ids") or []:
        block = blocks.get(layer_id)
        if block is None:
            raise KeyError(f"the page has no text block {layer_id}")
        if block.bounds is None:
            raise ValueError(f"text block {layer_id} has no frame; give a region instead")
        b = block.bounds
        boxes.append((b.x, b.y, b.x + b.width, b.y + b.height))
    if not boxes:
        raise ValueError("give a region or layer_ids, or use mode 'page'")
    return (
        math.floor(min(b[0] for b in boxes)),
        math.floor(min(b[1] for b in boxes)),
        math.ceil(max(b[2] for b in boxes)),
        math.ceil(max(b[3] for b in boxes)),
    )


@tool(
    "edit_image",
    "Redraw part of a page's art with an image model (the signed-in ChatGPT account's) and add the result as a new "
    "'edit' raster layer above the art. For lettering typesetting cannot redo, above all sound effects: give the "
    "region in page pixels, or the layer_ids of text blocks whose frames cover it, and an instruction saying exactly "
    "what to change, e.g. \"Translate the sound effect ドドド into English as RUMBLE, hand-lettered in the same style, "
    "size, angle and colour.\" Name the target language. Only the region changes; the model also sees a margin "
    "around it for context. Covered text blocks are hidden (hide_text) so typeset text does not draw over the edit. "
    "mode 'page' redraws the whole page, translating all of its lettering into the art and hiding every text block; "
    "the model draws at most 1536 pixels on the long side, so large pages come back softer. base 'source' (default) "
    "is the original scan with its original lettering; 'current' is the page as edited (cleanup, paint, earlier "
    "edits) without text. Takes up to a few minutes and uses the ChatGPT plan's image limit; one undo removes it.",
    {
        "page_id": _PAGE_ID,
        "instruction": {"type": "string"},
        "mode": {"type": "string", "enum": ["region", "page"]},
        "region": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}},
            "required": ["x", "y", "width", "height"],
            "additionalProperties": False,
        },
        "layer_ids": _LAYER_IDS,
        "base": {"type": "string", "enum": ["source", "current"]},
        "hide_text": {"type": "boolean", "description": "Hide the covered text blocks (default true)."},
        "quality": {"type": "string", "enum": list(QUALITIES)},
    },
    required=("instruction",),
    mutates=True,
)
async def edit_image(context: ToolContext, args: dict[str, Any]) -> Any:
    project, page_id = context.project(), context.page(args.get("page_id"))
    if context.image_editor is None:
        raise ValueError("no image model is available; sign in with ChatGPT")
    instruction = str(args["instruction"]).strip()
    if not instruction:
        raise ValueError("the instruction is empty")
    page = context.workspace.page(project, page_id)
    width, height = int(page.width), int(page.height)
    hide_text = args.get("hide_text", True)
    whole = (args.get("mode") or "region") == "page"
    hide: tuple[str, ...] = ()
    if whole:
        region = None
        scope, focus = "This is a whole manga page.", ""
    else:
        left, top, right, bottom = _region_box(args, {b.layer_id: b for b in page.blocks})
        left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
        if right - left < 4 or bottom - top < 4:
            raise ValueError(f"the region is outside the page or too small (the page is {width}x{height})")
        region = Box(left, top, right, bottom)
        hide = tuple(args.get("layer_ids") or []) if hide_text else ()
        scope, focus = "This is a crop of a manga page.", ", in the middle of the image; the edges are only context"
    art = await asyncio.to_thread(context.workspace.page_art, project, page_id, args.get("base") or "source")
    prompt = _EDIT_PROMPT.format(scope=scope, instruction=instruction, focus=focus)
    layer, window, size = await edit_region(context.image_editor, art, region, prompt, args.get("quality") or "auto")
    name = "Edit: " + (instruction if len(instruction) <= 40 else instruction[:39].rstrip() + "…")
    view, layer_id, hidden = context.workspace.add_raster_layer(
        project, page_id, name, layer, hide_layers=hide, hide_all_text=whole and hide_text
    )
    box = region or Box(0, 0, width, height)
    return {
        "layer_id": layer_id,
        "page_id": page_id,
        "page": {"width": width, "height": height},
        "region": {"x": box.left, "y": box.top, "width": box.width, "height": box.height},
        "sent": {"x": window.left, "y": window.top, "width": window.width, "height": window.height, "size": f"{size[0]}x{size[1]}"},
        "hidden_text_layers": hidden,
        "revision": view.revision,
    }


@tool("undo", "Undo the last change to the open project.", mutates=True)
def undo(context: ToolContext, args: dict[str, Any]) -> Any:
    view = context.workspace.undo(context.project())
    return {"can_undo": view.can_undo, "can_redo": view.can_redo}


@tool("redo", "Redo the last undone change to the open project.", mutates=True)
def redo(context: ToolContext, args: dict[str, Any]) -> Any:
    view = context.workspace.redo(context.project())
    return {"can_undo": view.can_undo, "can_redo": view.can_redo}


# -- pipeline and settings ------------------------------------------------------------------------


@tool(
    "run_pipeline",
    "Run pipeline stages and wait for them to finish. mode: full (every stage), through (every stage up to "
    "and including `stage`), only (just `stage`), or stages (the listed `stages`). scope: page (the open page, "
    "or page_ids), chapter (the open page's chapter, or chapter_id), or project. Re-runs never overwrite text "
    "the user (or you) edited.",
    {
        "mode": {"type": "string", "enum": ["full", "through", "only", "stages"]},
        "stage": {"type": "string", "enum": STAGES},
        "stages": {"type": "array", "items": {"type": "string", "enum": STAGES}},
        "scope": {"type": "string", "enum": ["page", "chapter", "project"]},
        "page_ids": {"type": "array", "items": {"type": "string"}},
        "chapter_id": {"type": "string"},
    },
    required=("mode", "scope"),
    mutates=True,
)
async def run_pipeline(context: ToolContext, args: dict[str, Any]) -> Any:
    project = context.project()
    mode = args["mode"]
    if mode == "full":
        operation = Operation.full()
    elif mode in ("through", "only"):
        if not args.get("stage"):
            raise ValueError(f"mode {mode!r} needs a stage")
        stage = Stage.parse(args["stage"])
        operation = Operation.through(stage) if mode == "through" else Operation.only(stage)
    else:
        operation = Operation.select([Stage.parse(s) for s in args.get("stages") or []])
    scope = args["scope"]
    if scope == "page":
        page_ids: Optional[list[str]] = list(args.get("page_ids") or [context.page(None)])
    elif scope == "chapter":
        page_ids = list(_chapter_pages(context, args.get("chapter_id")))
    else:
        page_ids = None
    job = await context.workspace.start_run(project, operation, page_ids)
    try:
        async for progress in context.workspace.events(job.id):
            context.emit(
                {
                    "type": "progress",
                    "kind": progress.kind.value,
                    "page_id": progress.page,
                    "stage": progress.stage.value if progress.stage else None,
                    "model": progress.model,
                    "completed": progress.completed,
                    "total": progress.total,
                    "message": progress.message,
                }
            )
    except asyncio.CancelledError:
        # Stopping the assistant stops the run it started.
        context.workspace.stop(job.id)
        raise
    return {"status": job.status, "completed": job.completed, "total": job.total, "error": job.error}


@tool(
    "configure_translation",
    "Change translation settings: target language (a BCP 47 tag from get_translation_settings), custom "
    "instructions for the translator (style, glossary, honorifics; empty string clears them), or the "
    "provider and model (from list_translation_models).",
    {
        "target_language": {"type": "string"},
        "instructions": {"type": "string"},
        "provider": {"type": "string"},
        "model": {"type": "string"},
    },
    mutates=True,
)
def configure_translation(context: ToolContext, args: dict[str, Any]) -> Any:
    config = context.workspace.config.model_copy(deep=True)
    translation = config.translation
    if "target_language" in args:
        translation.target_language = args["target_language"]
    if "instructions" in args:
        translation.instructions = args["instructions"] or None
    if "provider" in args:
        translation.model.provider = args["provider"]
        translation.model.model = args.get("model")
    elif "model" in args:
        translation.model.model = args["model"]
    context.workspace.set_config(config)
    context.changed.add("config")
    return get_translation_settings.handler(context, {})


@tool("open_page", "Show a page in the editor so the user can see it.", {"page_id": {"type": "string"}}, required=("page_id",))
def open_page(context: ToolContext, args: dict[str, Any]) -> Any:
    page = context.workspace.page(context.project(), args["page_id"])
    context.page_id = page.id
    context.emit({"type": "ui", "action": "open_page", "page_id": page.id})
    return {"opened": page.id, "name": page.name}


APP_TOOLS: list[Tool] = [
    list_projects,
    get_project,
    get_page,
    search_text,
    get_translation_settings,
    list_translation_models,
    list_fonts,
    edit_text_blocks,
    set_typography,
    set_layer_visibility,
    create_text_block,
    delete_text_blocks,
    edit_image,
    undo,
    redo,
    run_pipeline,
    configure_translation,
    open_page,
]
