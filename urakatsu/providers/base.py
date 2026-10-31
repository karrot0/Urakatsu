"""Provider base classes with per-provider runtime selection.

A provider is one model (or remote service) that implements a stage. It
declares every runtime it can run on, in preference order; configuration picks
one per provider (``PipelineConfig.runtimes``), and any declared runtime whose
Python package is installed can be selected. Stage logic lives in
``urakatsu.pipeline.stages`` and is shared by all providers of a stage.

Lifecycle (the ``StageProcessor`` model cell): ``load()`` -> calls ... -> ``unload()``.
``load`` is idempotent and serialised; ``_load`` runs on a worker thread.

Remote (not in Koharu): detection, OCR and inpainting providers also declare the ``remote``
runtime, which runs the same provider on a Urakatsu worker on another machine
(``urakatsu.remote``). It is never picked by default. It runs on the app's worker picked in
the runtime menu (``remote_worker``; Settings, Workers; ``urakatsu.remote.workers``), else on
``remote_url``, else on the first of the app's workers that answers. Settings fields
that only apply to some runtimes name them in their schema (``"runtimes"``), and the Settings
page shows them only for those. The ``cloud`` runtime is the same client against Urakatsu
Cloud (``urakatsu.remote.cloud``): available once the account key is set, for the providers
the cloud hosts. A provider can also declare ``cloud`` as its only runtime (Nano Banana), for a
model only Urakatsu Cloud runs.
"""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.stage import Stage
from urakatsu.core.types import Availability
from urakatsu.remote import CLOUD, OFFLOADED, REMOTE
from urakatsu.runtimes.store import DownloadProgress, WeightSource

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.catalog.facts import ModelFacts
    from urakatsu.runtimes.base import Runtime
    from urakatsu.runtimes.manager import RuntimeManager


class ProviderSettings(BaseModel):
    """Default settings model. Providers subclass this for their options."""

    model_config = {"extra": "ignore", "protected_namespaces": ()}
    # An installed catalog model to use instead of the provider's default weights.
    model: Optional[str] = Field(default=None, description="Installed model (Models page); empty uses the default.")
    # Where the ``remote`` runtime finds its worker (``urakatsu worker``).
    remote_url: Optional[str] = Field(
        default=None,
        description="Address of a Urakatsu worker (urakatsu worker) that runs this model. Empty uses the first worker in Settings, Workers that answers.",
        json_schema_extra={"runtimes": [REMOTE]},
    )
    remote_api_key: Optional[str] = Field(
        default=None, description="The token the worker printed when it started.", json_schema_extra={"runtimes": [REMOTE]}
    )
    # Which of the app's workers runs it (Settings, Workers), picked in the runtime menu.
    remote_worker: Optional[str] = Field(default=None, json_schema_extra={"runtimes": [REMOTE]})


def server_fields(runtime: str) -> tuple[Any, Any]:
    """``server_url`` / ``server_api_key`` for a provider that runs as ``runtime``'s server
    (llama-server, sd-server): set, the provider uses that server instead of starting one."""
    url = Field(
        default=None,
        description="Address of a running server to use instead of starting one here. It must serve this model's weights.",
        json_schema_extra={"runtimes": [runtime]},
    )
    key = Field(default=None, description="Sent as a bearer token, if the server or its proxy asks for one.", json_schema_extra={"runtimes": [runtime]})
    return url, key


class RuntimeOption(BaseModel):
    name: str
    available: Availability


class ProviderInfo(BaseModel):
    id: str
    name: str
    stage: Stage
    description: str = ""
    runtimes: list[RuntimeOption] = Field(default_factory=list)
    default_runtime: Optional[str] = None
    weights: list[str] = Field(default_factory=list)
    available: Availability
    settings_schema: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    prompted: bool = False
    instructed: bool = False


class Provider(ABC):
    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    stage: ClassVar[Stage]
    description: ClassVar[str] = ""
    # Runtimes this provider can run on, preferred first. Empty for pure-Python / remote providers.
    runtimes: ClassVar[tuple[str, ...]] = ()
    # Importable modules each runtime path needs in addition to the runtime package itself.
    requires: ClassVar[dict[str, tuple[str, ...]]] = {}
    Settings: ClassVar[type[ProviderSettings]] = ProviderSettings
    # Role of each entry of ``weight_sources(runtime)``, per runtime, for GGUF / diffusion files that
    # an installed model replaces by role ("model", "projector", "diffusion_model"). Runtimes not
    # listed replace files by name (Transformers / ONNX repos keep their file names).
    weight_roles: ClassVar[dict[str, tuple[str, ...]]] = {}
    # Hugging Face searches that find models this provider accepts, run when the Models page
    # lists usable models without a query (they name families, never specific repositories).
    catalog_queries: ClassVar[tuple[str, ...]] = ()
    # Whether the provider can run on a Urakatsu worker (the ``remote`` runtime).
    remotable: ClassVar[bool] = False

    @classmethod
    def accepts(cls, facts: "ModelFacts") -> Optional[str]:
        """The runtime this provider would run a catalog model on, or None if it cannot use it."""
        return None

    def __init__(
        self,
        settings: dict[str, Any] | ProviderSettings | None = None,
        runtimes: "RuntimeManager | None" = None,
        runtime: Optional[str] = None,
    ):
        if isinstance(settings, ProviderSettings):
            self.settings = settings
        else:
            self.settings = self.Settings.model_validate(settings or {})
        self.runtimes = runtimes
        if runtime is not None and runtime not in self.runtimes_declared():
            raise ValueError(f"{self.id} cannot run on {runtime!r}; supported: {list(self.runtimes_declared())}")
        self.runtime_name: Optional[str] = runtime or self.default_runtime()
        self._loaded = False
        self._lock = asyncio.Lock()
        self._remote = None
        if self.runtime_name in OFFLOADED:
            from urakatsu.remote.client import RemoteModel

            # The stages call the inference methods directly (inpainting per tile), so the
            # worker's versions shadow the class's on this instance.
            self._remote = RemoteModel(self)
            for name, method in self._remote_methods(self._remote).items():
                setattr(self, name, method)

    def _remote_methods(self, remote) -> dict[str, Any]:
        """The inference methods a ``RemoteModel`` takes over; each stage interface names its own."""
        return {}

    # -- metadata -----------------------------------------------------------------------------

    @classmethod
    def runtimes_declared(cls) -> tuple[str, ...]:
        return (*cls.runtimes, REMOTE, CLOUD) if cls.remotable else tuple(cls.runtimes)

    @classmethod
    def runtime_availability(cls, runtime: str) -> Availability:
        from urakatsu.runtimes.manager import RuntimeManager

        if runtime in OFFLOADED and not cls.remotable and runtime not in cls.runtimes:
            return Availability.no(f"{cls.id} cannot run on a worker")
        if runtime == REMOTE:
            return Availability.yes()
        if runtime == CLOUD:
            from urakatsu.remote import cloud

            return cloud.availability(cls.id)

        base = RuntimeManager.availability(runtime)
        if not base.ok:
            return base
        missing = [m for m in cls.requires.get(runtime, ()) if importlib.util.find_spec(m) is None]
        if missing:
            return Availability.no(f"missing Python modules: {', '.join(missing)}")
        return Availability.yes()

    @classmethod
    def default_runtime(cls) -> Optional[str]:
        """The first available local runtime: ``remote`` and ``cloud`` are only ever chosen on purpose."""
        declared = tuple(cls.runtimes)
        for runtime in declared:
            if cls.runtime_availability(runtime).ok:
                return runtime
        return declared[0] if declared else None

    @classmethod
    def available(cls, runtime: Optional[str] = None) -> Availability:
        """Whether ``runtime`` can run; without one, whether a local runtime can."""
        declared = tuple(cls.runtimes)
        if runtime in OFFLOADED:
            return cls.runtime_availability(runtime)
        if not declared:
            missing = [m for m in cls.requires.get("", ()) if importlib.util.find_spec(m) is None]
            return Availability.no(f"missing Python modules: {', '.join(missing)}") if missing else Availability.yes()
        if runtime is not None:
            return cls.runtime_availability(runtime)
        reasons = []
        for name in declared:
            availability = cls.runtime_availability(name)
            if availability.ok:
                return availability
            reasons.append(f"{name}: {availability.reason}")
        return Availability.no("; ".join(reasons))

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        """Weights resolved before loading on ``runtime``. Override when they differ per runtime."""
        return ()

    @classmethod
    def uses_catalog(cls) -> bool:
        return cls.accepts.__func__ is not Provider.accepts.__func__

    @classmethod
    def info(cls) -> ProviderInfo:
        default = cls.default_runtime()
        schema = cls.Settings.model_json_schema()
        defaults = cls.Settings().model_dump(mode="json")
        hidden = ["remote_worker"] if cls.uses_catalog() else ["model", "remote_worker"]  # the runtime menu picks the worker
        if not cls.remotable:
            hidden += ["remote_url", "remote_api_key"]  # remote_worker is hidden above
        for name in hidden:
            schema.get("properties", {}).pop(name, None)
            defaults.pop(name, None)
        return ProviderInfo(
            id=cls.id,
            name=cls.name or cls.id,
            stage=Stage.parse(cls.stage),
            description=cls.description,
            runtimes=[RuntimeOption(name=r, available=cls.runtime_availability(r)) for r in cls.runtimes_declared()],
            default_runtime=default,
            weights=[w.describe() for w in cls.weight_sources(default)],
            available=cls.available(),
            settings_schema=schema,
            defaults=defaults,
            prompted=bool(getattr(cls, "prompted", False)),
            instructed=bool(getattr(cls, "instructed", False)),
        )

    # -- lifecycle -----------------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            availability = Availability.yes() if self.attached() else self.available(self.runtime_name)
            if not availability.ok:
                raise ProviderUnavailable(f"{self.id}: {availability.reason}")
            await asyncio.to_thread(self._remote.load if self._remote is not None else self._load)
            self._loaded = True

    def attached(self) -> bool:
        """Whether the provider uses a server someone else runs (``server_url`` set, on the runtime
        that field is for), so it needs no local runtime and no local weights."""
        field = type(self.settings).model_fields.get("server_url")
        if field is None or not (getattr(self.settings, "server_url", None) or "").strip():
            return False
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        return self.runtime_name in extra.get("runtimes", ())

    def _load(self) -> None:
        """Synchronous hook: resolve weights and build the model on ``self.runtime_name``."""

    def unload(self) -> bool:
        if not self._loaded:
            return False
        if self._remote is not None:
            self._remote.unload()
        else:
            self._unload()
        self._loaded = False
        return True

    def _unload(self) -> None:
        """Synchronous hook: drop references so memory is reclaimed."""

    # -- helpers -------------------------------------------------------------------------------

    def store(self):
        if self.runtimes is None:
            from urakatsu.runtimes.store import ModelStore

            return ModelStore()
        return self.runtimes.store

    def resolve(self, source: WeightSource, progress: Optional[DownloadProgress] = None) -> Path:
        return source.resolve(self.store(), progress)

    def resolve_weights(self, progress: Optional[DownloadProgress] = None) -> list[Path]:
        return [self.resolve(w, progress) for w in self.selected_sources(self.weight_sources(self.runtime_name))]

    def selected_sources(self, sources: tuple[WeightSource, ...]) -> tuple[WeightSource, ...]:
        """``sources`` with the files of the installed model chosen in settings swapped in."""
        model_id = getattr(self.settings, "model", None)
        if not model_id:
            return sources
        from urakatsu.catalog.installed import InstalledModels

        installed = InstalledModels(self.store()).get(model_id)
        if installed is None:
            raise ProviderUnavailable(f"{self.id}: installed model {model_id!r} was removed; pick another on the Models page")
        roles = self.weight_roles.get(self.runtime_name or "")
        replaced = []
        for index, source in enumerate(sources):
            if roles is not None and index < len(roles):
                replacement = installed.source_for_role(roles[index])
            else:
                name = getattr(source, "filename", "")
                replacement = installed.source_for_name(name)
                weights = [f for f in installed.files if f.path.endswith(".safetensors")]
                if replacement is None and name.endswith(".safetensors") and len(weights) == 1:
                    replacement = installed.source(weights[0].path)  # one weights file under another name
            replaced.append(replacement or source)
        return tuple(replaced)

    def runtime_instance(self) -> "Runtime":
        if not self.runtime_name:
            raise ProviderUnavailable(f"{self.id} does not run on a runtime")
        if self.runtimes is None:
            from urakatsu.runtimes.manager import RuntimeManager

            self.runtimes = RuntimeManager()
        return self.runtimes.runtime(self.runtime_name)


# ---- stage interfaces ----------------------------------------------------------------------------


class DetectionProvider(Provider):
    """Returns ``urakatsu.core.results.LayoutDetections``."""

    stage = Stage.DETECTION
    remotable = True

    def _remote_methods(self, remote) -> dict[str, Any]:
        return {"_detect": remote.detect}

    async def detect(self, image: Image.Image):
        return await asyncio.to_thread(self._detect, image)

    def _detect(self, image: Image.Image):  # pragma: no cover - interface
        raise NotImplementedError


class OcrProvider(Provider):
    """Reads text from one region crop at a time."""

    stage = Stage.OCR
    remotable = True

    def _remote_methods(self, remote) -> dict[str, Any]:
        return {"_recognize": remote.recognize}

    async def recognize(self, crops: list[Image.Image]) -> list[str]:
        return await asyncio.to_thread(self._recognize, crops)

    def _recognize(self, crops: list[Image.Image]) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class InpaintingProvider(Provider):
    """Inpaints ``image`` under ``mask`` (HxW uint8, non-zero = remove). Stage logic decides tiling."""

    stage = Stage.INPAINTING
    # Upstream runs LaMa / AOT through ``inpaint_tiled`` and diffusion models on whole regions.
    tiled: ClassVar[bool] = True
    # Whether ``_fill`` follows a prompt (diffusion models); generative fill needs one for prompts.
    prompted: ClassVar[bool] = False
    # Whether the prompt is an edit instruction ("replace X with Y"), as for models that take the
    # crop as a reference image, rather than a description of what belongs there.
    instructed: ClassVar[bool] = False
    # Whether the model edits the whole crop it gets from the instruction, not knowing the mask
    # (it takes the crop as a reference image); generative fill then sends a tight crop.
    crop_edits: ClassVar[bool] = False
    remotable = True

    def _remote_methods(self, remote) -> dict[str, Any]:
        methods: dict[str, Any] = {"_inpaint": remote.inpaint}
        if self.prompted:  # the others fill by inpainting, which already goes to the worker
            methods["_fill"] = remote.fill
        return methods

    async def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        return await asyncio.to_thread(self._inpaint, image, mask)

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:  # pragma: no cover
        raise NotImplementedError

    def _fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        """Generative fill of ``mask``: ``prompt`` describes what to paint there; empty removes what is under it."""
        if prompt and not self.prompted:
            raise ValueError(f"{self.name} cannot follow a prompt; leave it empty to remove, or pick a diffusion model for Generative fill in Settings, Pipeline")
        return self._inpaint(image, mask)


class WatermarkBox(BaseModel):
    box: tuple[int, int, int, int]  # left, top, right, bottom
    score: float


class WatermarkProvider(Provider):
    """Proposes boxes that may hold a watermark; the watermarks stage decides what is removed."""

    stage = Stage.WATERMARKS

    async def detect_watermarks(self, image: Image.Image) -> list[WatermarkBox]:
        return await asyncio.to_thread(self._detect_watermarks, image)

    def _detect_watermarks(self, image: Image.Image) -> list[WatermarkBox]:  # pragma: no cover - interface
        raise NotImplementedError


class TranslationProvider(Provider):
    """Translates segments."""

    stage = Stage.TRANSLATION
    # Estimated tokens of glossary, notes and context a request may carry (pipeline/knowledge.py).
    context_budget: ClassVar[int] = 6000

    async def translate(self, request) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError
