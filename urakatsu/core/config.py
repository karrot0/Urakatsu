"""Configuration, shaped like upstream's ``pipeline.toml``, ``providers.toml`` and ``typesetting.toml``.

- ``detection`` / ``ocr`` / ``inpainting`` select one model each; ``translation`` selects
  a provider + model (``ModelSelection``) with generation settings.
- ``fill`` selects the inpainting model the canvas's generative fill runs (not an upstream
  setting); empty uses the ``inpainting`` model.
- ``processor`` keeps every vision model's profile independently of the active model,
  so switching models keeps the other profiles (``ProcessorConfig``).
- ``providers`` holds the translation providers' settings (API keys, base URLs).
- ``runtimes`` selects the runtime for any provider that supports more than one.
- ``typesetting`` holds the renderer's font family list (``TypesettingConfig``).
- ``storage`` holds how page images are stored (``StorageConfig``; not an upstream setting).
- ``sound_effects`` says what happens to detected onomatopoeia (``SoundEffectsConfig``; not an
  upstream setting: Koharu keeps them as regions that nothing downstream uses).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Optional

import tomli_w
from pydantic import BaseModel, Field

from .stage import Stage


def home() -> Path:
    """URAKATSU_HOME, else the data folder setup saved on macOS and Linux, else ``~/.urakatsu``.

    On Windows setup saves the folder as the user's URAKATSU_HOME; macOS and Linux keep no
    user environment for apps started from the desktop, so setup writes it to
    ``~/.config/urakatsu/home`` (packaging/launcher/src/platform/unix.rs).
    """
    root = os.environ.get("URAKATSU_HOME")
    if root:
        return Path(root)
    if os.name != "nt":
        config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        try:
            saved = (config / "urakatsu" / "home").read_text(encoding="utf-8").strip()
        except OSError:
            saved = ""
        if saved:
            return Path(saved)
    return Path.home() / ".urakatsu"


class ModelSelection(BaseModel):
    """``ModelSelection``."""

    provider: str = "local"
    model: Optional[str] = "gemma4-12b-it"  # local::DEFAULT_MODEL
    quantization: Optional[str] = "Q4_K_XL"  # local::DEFAULT_QUANTIZATION
    vision: bool = True
    reasoning: bool = True


class GenerationConfig(BaseModel):
    """``GenerationConfig``."""

    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    repeat_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    reasoning: Optional[bool] = False
    vision: Optional[bool] = True

    def for_model(self, model: ModelSelection) -> "GenerationConfig":
        return self.model_copy(
            update={
                "reasoning": self.reasoning if model.reasoning else None,
                "vision": self.vision if model.vision else None,
            }
        )


class TranslationConfig(BaseModel):
    """``TranslationConfig``."""

    model: ModelSelection = Field(default_factory=ModelSelection)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    target_language: str = "en-US"
    instructions: Optional[str] = None


class ModelChoice(BaseModel):
    model: str


class TypesettingConfig(BaseModel):
    """``TypesettingConfig``."""

    font_families: list[str] = Field(default_factory=lambda: ["CCWildWords", "Adobe 黑体 Std"])


class StorageConfig(BaseModel):
    """How imported and downloaded page images are stored.

    Pages are kept as they arrive (JPEG, PNG, WebP). With ``lossless_pages``,
    a page that would be stored as PNG is re-encoded losslessly (same pixels) when that is
    smaller; JPEG and WebP pages stay as they are, since re-encoding them would grow them.
    """

    lossless_pages: bool = False


SOUND_EFFECT_MODES = ("off", "typeset", "redraw")


class SoundEffectsConfig(BaseModel):
    """What happens to detected sound effects.

    ``off`` keeps Koharu's behaviour. ``typeset`` turns each one into a text block that is read,
    translated, erased and lettered like any other. ``redraw`` reads and translates them too, then
    the sound effects stage has an image model redraw each one in the target language, in its
    original style (``pipeline/stages/sound_effects.py``). ``model`` is that image model: an
    inpainting provider that follows prompts; empty uses the generative fill model.
    """

    mode: Literal["off", "typeset", "redraw"] = "off"
    model: str = ""


WATERMARK_MODES = ("off", "remove")


class WatermarksConfig(BaseModel):
    """Repeated watermarks (platform logos, scanlator marks). Not in Koharu.

    ``off`` keeps Koharu's behaviour. ``remove`` runs the watermarks stage first: ``model`` (a
    watermark detector) proposes marks, and the ones that repeat across a chapter are removed by
    unblending them (``pipeline/stages/watermarks.py``).
    """

    mode: Literal["off", "remove"] = "off"
    model: str = "yolo26-watermark"


class PipelineConfig(BaseModel):
    device: str = "auto"  # auto | cpu | cuda | cuda:N | metal
    # Loaded models are unloaded after this many minutes without a run; 0 keeps them loaded.
    idle_unload_minutes: int = Field(default=15, ge=0)
    detection: ModelChoice = Field(default_factory=lambda: ModelChoice(model="koharu-layout-rfdetr-seg-2xl"))
    ocr: ModelChoice = Field(default_factory=lambda: ModelChoice(model="paddleocr-vl-1.6"))
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    inpainting: ModelChoice = Field(default_factory=lambda: ModelChoice(model="lama"))
    fill: ModelChoice = Field(default_factory=lambda: ModelChoice(model=""))
    processor: dict[str, dict[str, Any]] = Field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    runtimes: dict[str, str] = Field(default_factory=dict)
    typesetting: TypesettingConfig = Field(default_factory=TypesettingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sound_effects: SoundEffectsConfig = Field(default_factory=SoundEffectsConfig)
    watermarks: WatermarksConfig = Field(default_factory=WatermarksConfig)

    def provider_for(self, stage: Stage) -> str:
        if stage is Stage.TRANSLATION:
            return self.translation.model.provider
        if stage is Stage.SOUND_EFFECTS:
            return self.redraw_provider()
        if stage is Stage.WATERMARKS:
            return self.watermarks.model
        return {Stage.DETECTION: self.detection, Stage.OCR: self.ocr, Stage.INPAINTING: self.inpainting}[stage].model

    def fill_provider(self) -> str:
        """The inpainting provider generative fill runs: ``fill``, else the inpainting model."""
        return self.fill.model or self.inpainting.model

    def redraw_provider(self) -> str:
        """The inpainting provider that redraws sound effects: ``sound_effects.model``, else the fill model."""
        return self.sound_effects.model or self.fill_provider()

    def set_provider(self, stage: Stage, provider: str) -> None:
        if stage is Stage.TRANSLATION:
            self.translation.model.provider = provider
        elif stage is Stage.WATERMARKS:
            self.watermarks.model = provider
        else:
            {Stage.DETECTION: self.detection, Stage.OCR: self.ocr, Stage.INPAINTING: self.inpainting}[stage].model = provider

    def runtime_for(self, provider: str) -> Optional[str]:
        return self.runtimes.get(provider)

    def settings_for(self, stage: Stage, provider: str) -> dict[str, Any]:
        """Settings handed to the provider constructor.

        Translation providers get their service settings plus the active translation
        selection; vision models get their processor profile.
        """
        if stage is Stage.TRANSLATION:
            settings = dict(self.providers.get(provider, {}))
            settings["translation"] = self.translation.model_dump(mode="json")
            return settings
        return dict(self.processor.get(provider, {}))


def config_path() -> Path:
    return home() / "pipeline.toml"


def load_config(path: Path | None = None) -> PipelineConfig:
    path = path or config_path()
    if not path.exists():
        return PipelineConfig()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return PipelineConfig.model_validate(data)


def save_config(config: PipelineConfig, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(_strip_none(config.model_dump(mode="json"))), encoding="utf-8")
    return path


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value
