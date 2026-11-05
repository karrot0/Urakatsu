"""Tools for the pipeline's stages and models: which provider runs each stage, on which
runtime, with which installed model, and the Hugging Face catalog behind them.

Reading is ``read`` access. Anything that changes the app beyond the open project needs
``full`` access: switching a stage's provider, runtime, mode or settings, installing a
model (a download of up to many gigabytes), and unloading models.

Settings changes go through ``Workspace.set_config`` like the Settings page's, so they
are validated the same way and the UI reloads them. A model is chosen the way the Models
page's "use" button chooses it: an installed model's id goes in the provider's processor
profile (``processor[provider].model``), or in the translation selection.

The catalog is the HTTP routes' instance (``service.catalog_api.catalog``), so a download
the agent starts shows in the Models page's queue.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from urakatsu.core.config import PipelineConfig
from urakatsu.core.registry import registry
from urakatsu.core.stage import Stage

from .tools import Tool, ToolContext, tool

STAGES = [s.value for s in Stage.all()]
# The generative fill model: an inpainting provider that is not a stage of its own.
FILL = "fill"
CATALOG_RUNTIMES = ("llama", "diffusion", "torch", "onnx")

# Stages a run includes only in some modes: (config section, the stage's modes).
_MODES: dict[Stage, tuple[str, tuple[str, ...]]] = {
    Stage.WATERMARKS: ("watermarks", ("off", "remove")),
    Stage.CENSOR_DETECTION: ("censorship", ("off", "detect", "remove")),
    Stage.DECENSOR: ("censorship", ("off", "detect", "remove")),
    Stage.COLORIZATION: ("colorization", ("off", "colorize")),
    Stage.SOUND_EFFECTS: ("sound_effects", ("off", "typeset", "redraw")),
}
_STAGE_NOTES = {
    Stage.WATERMARKS: "removes repeated watermarks before detection (mode remove)",
    Stage.CENSOR_DETECTION: "maps censor bars into a mask (mode detect or remove); shares its mode with decensor",
    Stage.DETECTION: "finds text regions and balloons",
    Stage.OCR: "reads the source text",
    Stage.TRANSLATION: "translates the text",
    Stage.INPAINTING: "removes the original lettering",
    Stage.DECENSOR: "reconstructs the art under censor bars (mode remove); shares its mode with censor_detection",
    Stage.COLORIZATION: "paints black-and-white pages in color (mode colorize)",
    Stage.SOUND_EFFECTS: "typesets (mode typeset) or redraws (mode redraw) sound effects; its model is an inpainting provider that follows prompts, empty for the fill model",
}


def _catalog():
    from urakatsu.service.catalog_api import catalog

    return catalog()


def _stage(value: str) -> Optional[Stage]:
    """The stage ``value`` names; ``None`` for the generative fill model."""
    return None if value == FILL else Stage.parse(value)


def _provider_stage(stage: Optional[Stage]) -> Stage:
    """The stage whose providers can serve ``stage`` (fill and sound effects use inpainting providers)."""
    return Stage.INPAINTING if stage is None or stage is Stage.SOUND_EFFECTS else stage


def _selected(config: PipelineConfig, stage: Optional[Stage]) -> str:
    if stage is None:
        return config.fill_provider()
    return config.provider_for(stage)


def _processor_model(config: PipelineConfig, provider: str) -> Optional[str]:
    return (config.processor.get(provider) or {}).get("model") or None


def _installed() -> list[dict[str, Any]]:
    try:
        return _catalog().list_installed()
    except Exception:  # noqa: BLE001 - the stage list still answers without the catalog
        return []


def _installed_for(installed: list[dict[str, Any]], stage: Stage, provider: str) -> list[dict[str, Any]]:
    rows = []
    for model in installed:
        for usage in model.get("usable_by") or []:
            if usage.get("provider") == provider and usage.get("stage") == stage.value:
                rows.append(
                    {
                        "id": model["id"],
                        "runtime": usage.get("runtime"),
                        "size": model.get("size"),
                        "complete": model.get("complete", True),
                        "compatibility": (model.get("compatibility") or {}).get("status"),
                    }
                )
    return rows


def _stage_row(context: ToolContext, stage: Optional[Stage], loaded: dict[Stage, str]) -> dict[str, Any]:
    config = context.workspace.config
    provider = _selected(config, stage)
    row: dict[str, Any] = {"stage": stage.value if stage else FILL}
    if stage is not None:
        row["does"] = _STAGE_NOTES.get(stage, "")
    else:
        row["does"] = "the generative fill model the canvas and sound effect redraws use; empty provider follows inpainting"
    if stage in _MODES:
        section, _ = _MODES[stage]
        row["mode"] = getattr(config, section).mode
    row["provider"] = provider
    try:
        cls = registry.get(_provider_stage(stage), provider)
        row["provider_name"] = cls.name or cls.id
        row["available"] = cls.available().ok
    except KeyError:
        row["available"] = False
        row["problem"] = f"no provider {provider!r} is registered for this stage"
    row["runtime"] = config.runtime_for(provider) or "default"
    if stage is Stage.TRANSLATION:
        row["model"] = config.translation.model.model
        row["target_language"] = config.translation.target_language
    elif _processor_model(config, provider):
        row["model"] = _processor_model(config, provider)
    if stage is not None and loaded.get(stage):
        row["loaded"] = True
    return row


# -- reading --------------------------------------------------------------------------------------


@tool(
    "get_pipeline",
    "Every pipeline stage in run order (plus 'fill', the generative fill model) with what it does, its mode for "
    "the optional stages, the provider that runs it, the runtime, the installed model it uses when one is chosen, "
    "and whether it is loaded now. Also the device.",
)
def get_pipeline(context: ToolContext, args: dict[str, Any]) -> Any:
    loaded = context.workspace.loaded()
    stages = [_stage_row(context, stage, loaded) for stage in Stage.all()]
    stages.append(_stage_row(context, None, loaded))
    return {"device": context.workspace.config.device, "busy": context.workspace.pipeline.busy, "stages": stages}


@tool(
    "list_stage_providers",
    "The providers that can run one stage (or 'fill'): id, name, description, whether it is available here and "
    "why not, its runtimes, its settings and their current values, and the installed catalog models it can use "
    "(choose one with configure_stage's model).",
    {"stage": {"type": "string", "enum": [*STAGES, FILL]}},
    required=("stage",),
)
async def list_stage_providers(context: ToolContext, args: dict[str, Any]) -> Any:
    stage = _stage(args["stage"])
    config = context.workspace.config
    registry.sync(config.providers)
    selected = _selected(config, stage)
    installed = await asyncio.to_thread(_installed)
    providers = []
    for cls in registry.list(_provider_stage(stage)):
        info = cls.info()
        row: dict[str, Any] = {
            "id": info.id,
            "name": info.name,
            "selected": info.id == selected,
            "description": info.description,
            "available": info.available.ok,
            "runtimes": [{"name": r.name, "available": r.available.ok} for r in info.runtimes],
            "default_runtime": info.default_runtime,
        }
        if not info.available.ok:
            row["unavailable_reason"] = info.available.reason
        if config.runtime_for(info.id):
            row["runtime"] = config.runtime_for(info.id)
        if stage is not Stage.TRANSLATION:
            properties = info.settings_schema.get("properties") or {}
            if properties:
                current = {**info.defaults, **(config.processor.get(info.id) or {})}
                row["settings"] = {
                    name: {
                        "value": current.get(name),
                        **{k: v for k, v in schema.items() if k in ("type", "enum", "minimum", "maximum", "description")},
                    }
                    for name, schema in properties.items()
                }
            models = _installed_for(installed, _provider_stage(stage), info.id)
            if models:
                row["installed_models"] = models
                row["model"] = _processor_model(config, info.id)
        else:
            models = _installed_for(installed, Stage.TRANSLATION, info.id)
            if models:
                row["installed_models"] = models
        providers.append(row)
    return {"stage": args["stage"], "providers": providers}


# -- changing stages ------------------------------------------------------------------------------


@tool(
    "configure_stage",
    "Change how one stage (or 'fill') runs. provider: a provider id from list_stage_providers. runtime: one of the "
    "provider's runtimes, or 'default' for the first available. model: an installed model id from "
    "list_stage_providers (for translation also a hosted model id from list_translation_models). mode: for the "
    "optional stages (watermarks off/remove, censor_detection and decensor off/detect/remove, colorization "
    "off/colorize, sound_effects off/typeset/redraw). settings: the provider's own settings to change (names from "
    "list_stage_providers; not for translation). The change applies to the next run; loaded models reload.",
    {
        "stage": {"type": "string", "enum": [*STAGES, FILL]},
        "provider": {"type": "string"},
        "runtime": {"type": "string"},
        "model": {"type": "string"},
        "mode": {"type": "string"},
        "settings": {"type": "object", "description": "Setting name to new value; null restores the default."},
    },
    required=("stage",),
    mutates=True,
    access="full",
)
async def configure_stage(context: ToolContext, args: dict[str, Any]) -> Any:
    stage = _stage(args["stage"])
    config = context.workspace.config.model_copy(deep=True)
    registry.sync(config.providers)
    changed: list[str] = []

    if args.get("mode") is not None:
        if stage not in _MODES:
            raise ValueError(f"{args['stage']} has no modes; only {', '.join(s.value for s in _MODES)} do")
        section, modes = _MODES[stage]
        if args["mode"] not in modes:
            raise ValueError(f"{args['stage']} mode must be one of {list(modes)}")
        getattr(config, section).mode = args["mode"]
        changed.append("mode")

    provider = args.get("provider")
    if provider is not None:
        if provider or stage not in (None, Stage.SOUND_EFFECTS):
            registry.get(_provider_stage(stage), provider)  # raises with the known ids
        if stage is None:
            config.fill.model = provider
        elif stage is Stage.SOUND_EFFECTS:
            config.sound_effects.model = provider
        else:
            config.set_provider(stage, provider)
        changed.append("provider")
    provider = _selected(config, stage)
    cls = registry.get(_provider_stage(stage), provider)

    if args.get("runtime") is not None:
        runtime = args["runtime"]
        if runtime in ("", "default"):
            config.runtimes.pop(provider, None)
        else:
            declared = cls.runtimes_declared()
            if runtime not in declared:
                raise ValueError(f"{provider} cannot run on {runtime!r}; it runs on {list(declared)}")
            availability = cls.runtime_availability(runtime)
            if not availability.ok:
                raise ValueError(f"{runtime} is not available for {provider}: {availability.reason}")
            config.runtimes[provider] = runtime
        changed.append("runtime")

    if args.get("model") is not None:
        _choose_model(config, stage, provider, cls, str(args["model"]))
        changed.append("model")

    if args.get("settings"):
        if stage is Stage.TRANSLATION:
            raise ValueError("translation settings change through configure_translation (or the Settings page for API keys)")
        schema, defaults = cls._settings_info()
        allowed = set(schema.get("properties") or {})
        unknown = sorted(set(args["settings"]) - allowed)
        if unknown:
            raise ValueError(f"{provider} has no settings {unknown}; it has {sorted(allowed)}")
        current = dict(config.processor.get(provider) or {})
        for name, value in args["settings"].items():
            if value is None:
                current.pop(name, None)
            else:
                current[name] = value
        cls.Settings.model_validate({**defaults, **current})  # rejects values of the wrong type or range
        config.processor[provider] = current
        changed.append("settings")

    if not changed:
        raise ValueError("nothing to change; give a provider, runtime, model, mode or settings")
    await asyncio.to_thread(context.workspace.set_config, config)
    context.changed.add("config")
    return {"changed": changed, **_stage_row(context, stage, context.workspace.loaded())}


def _choose_model(config: PipelineConfig, stage: Optional[Stage], provider: str, cls, model: str) -> None:
    installed = {m["id"]: m for m in _installed()}
    found = installed.get(model)
    if stage is Stage.TRANSLATION:
        selection = config.translation.model
        if found is not None:
            usable = [u for u in found.get("usable_by") or [] if u.get("stage") == "translation"]
            if usable and provider not in {u["provider"] for u in usable}:
                selection.provider = usable[0]["provider"]  # the installed model says which provider runs it
            selection.model = found["id"]
            selection.quantization = found.get("variant") or "default"
            selection.vision = any(f.get("role") == "projector" for f in found.get("files") or [])
            selection.reasoning = False
        else:
            selection.model = model or None
        return
    if not cls.uses_catalog():
        raise ValueError(f"{provider} runs its own built-in weights; it takes no installed model")
    if model == "":
        settings = dict(config.processor.get(provider) or {})
        settings.pop("model", None)
        config.processor[provider] = settings
        return
    if found is None:
        raise KeyError(f"no installed model {model!r}; install it first (install_model) or pick from list_stage_providers")
    if found.get("complete") is False:
        raise ValueError(f"{model} is still downloading")
    stage_name = _provider_stage(stage).value
    if not any(u.get("provider") == provider and u.get("stage") == stage_name for u in found.get("usable_by") or []):
        users = ", ".join(f"{u['stage']}/{u['provider']}" for u in found.get("usable_by") or []) or "nothing"
        raise ValueError(f"{provider} cannot use {model}; it is usable by {users}")
    config.processor[provider] = {**(config.processor.get(provider) or {}), "model": model}


@tool(
    "unload_models",
    "Free GPU and system memory by unloading one stage's model, or every loaded model; they load again when next "
    "needed. Refused while a run is using them.",
    {"stage": {"type": "string", "enum": STAGES}},
    mutates=True,
    access="full",
)
async def unload_models(context: ToolContext, args: dict[str, Any]) -> Any:
    return {"unloaded": await context.workspace.unload_models(args.get("stage"))}


@tool("get_memory", "GPU and system memory: totals, what is free, Urakatsu's share, and which stage models are loaded.")
def get_memory(context: ToolContext, args: dict[str, Any]) -> Any:
    return context.workspace.memory()


# -- the model catalog ----------------------------------------------------------------------------

_RUNTIME = {
    "type": "string",
    "enum": list(CATALOG_RUNTIMES),
    "description": "llama: GGUF language and vision-language models (translation, OCR); diffusion: image models "
    "(inpainting, fill, sound effects, decensor, colorization); torch / onnx: other vision models (detection, OCR, "
    "inpainting).",
}


@tool(
    "search_models",
    "Search the Hugging Face catalog for models a stage can use, rated for this machine. Each result says which "
    "stages and providers can use it, and whether it is installed. Read get_model_details before installing.",
    {"runtime": _RUNTIME, "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}},
    required=("runtime",),
)
async def search_models(context: ToolContext, args: dict[str, Any]) -> Any:
    found = await asyncio.to_thread(_catalog().search, args["runtime"], args.get("query") or "", limit=int(args.get("limit") or 15), usable_only=True)
    return [
        {
            "repo": r["repo"],
            "downloads": r["downloads"],
            "updated": r["updated"],
            "parameters": r.get("parameters"),
            "gated": r.get("gated"),
            "usable_by": [f"{u['stage']}/{u['provider']}" for u in r["usable_by"]],
            "installed": r["installed"],
            "default": r["default"],
            **({"fit": r["estimate"]["level"]} if r.get("estimate") else {}),
        }
        for r in found
    ]


@tool(
    "get_model_details",
    "One catalog repo: which stages can use it, whether this machine's runtime supports it, and its variants "
    "(quantizations) with size and how each fits in GPU and system memory; `recommended` is the best fit. The "
    "readme is cut short.",
    {"runtime": _RUNTIME, "repo": {"type": "string", "description": "owner/name"}},
    required=("runtime", "repo"),
)
async def get_model_details(context: ToolContext, args: dict[str, Any]) -> Any:
    details = await asyncio.to_thread(_catalog().details, args["runtime"], args["repo"])
    return {
        "repo": details["repo"],
        "gated": details.get("gated"),
        "architecture": details.get("architecture"),
        "parameters": details.get("parameters"),
        "vision": details.get("vision"),
        "usable_by": details["usable_by"],
        "compatibility": details.get("compatibility"),
        "missing": details.get("missing"),
        "recommended": details.get("recommended"),
        "variants": [{"id": v["id"], "size": v["size"], "fit": v.get("fit"), "recommended": v.get("recommended")} for v in details["variants"]],
        "installed": [m["id"] for m in details.get("installed") or []],
        "readme": (details.get("readme") or "")[:3000],
    }


@tool(
    "install_model",
    "Download a catalog model variant (from get_model_details) into the models folder. It queues behind other "
    "downloads and runs in the background; follow it with list_downloads, then choose it with configure_stage. "
    "Downloads can be many gigabytes: say the size before starting one.",
    {"runtime": _RUNTIME, "repo": {"type": "string"}, "variant": {"type": "string"}},
    required=("runtime", "repo", "variant"),
    mutates=True,
    access="full",
)
async def install_model(context: ToolContext, args: dict[str, Any]) -> Any:
    download = await asyncio.to_thread(_catalog().install, args["runtime"], args["repo"], args["variant"])
    context.changed.add("models")
    return _download(download)


def _download(d: dict[str, Any]) -> dict[str, Any]:
    model = d.get("model") or {}
    return {
        "id": d["id"],
        "model": model.get("id"),
        "status": d["status"],
        "position": d.get("position"),
        "done": d.get("done"),
        "total": d.get("total"),
        "error": d.get("error"),
        "usable_by": [f"{u['stage']}/{u['provider']}" for u in model.get("usable_by") or []],
    }


@tool("list_downloads", "Model downloads: the running one, the queue, paused ones, and those finished in the last hour.")
async def list_downloads(context: ToolContext, args: dict[str, Any]) -> Any:
    return [_download(d) for d in await asyncio.to_thread(_catalog().list_downloads)]


@tool(
    "list_installed_models",
    "Models installed from the catalog: id, runtime, size, which stages and providers can use each, whether its "
    "files are complete, and whether this machine's runtime supports it.",
)
async def list_installed_models(context: ToolContext, args: dict[str, Any]) -> Any:
    return [
        {
            "id": m["id"],
            "runtime": m["runtime"],
            "size": m.get("size"),
            "complete": m.get("complete", True),
            "compatibility": (m.get("compatibility") or {}).get("status"),
            "usable_by": [f"{u['stage']}/{u['provider']}" for u in m.get("usable_by") or []],
        }
        for m in await asyncio.to_thread(_installed)
    ]


STAGE_TOOLS: list[Tool] = [
    get_pipeline,
    list_stage_providers,
    configure_stage,
    get_memory,
    unload_models,
    search_models,
    get_model_details,
    install_model,
    list_downloads,
    list_installed_models,
]
