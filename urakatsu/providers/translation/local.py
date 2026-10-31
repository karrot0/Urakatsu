"""On-device GGUF translation.

Model: one catalog entry (``catalog.py``) plus its quantization, served by the newest
llama.cpp release (``llama-server``) on the ``llama`` runtime. Behaviour matches upstream:

- the prompt is the GGUF chat template rendered with the shared system prompt and the
  JSON payload; ``enable_thinking`` is passed only when the request enables reasoning
  and the model supports it;
- the reply is constrained to the translation JSON schema (llama.cpp converts it to a
  grammar);
- generation defaults come from the catalog entry and are overridden by the
  ``GenerationConfig`` (``ModelGeneration.options``);
- a page image is attached only when the model has a vision projector.

Divergences, all in the inference plumbing:
- Structured output uses llama.cpp's JSON-schema grammar, not llguidance, so the
  ``x-guidance`` whitespace options upstream sets are not sent.
- The image is sent as PNG (lossless) instead of raw pixels.
- The context window is fixed at ``N_CTX`` tokens when the model loads, instead of being
  sized per request; a prompt that does not fit fails the request.
- Chat templates also get ``strftime_now``, as in Transformers and llama.cpp; Mistral-style
  templates call it to date their default system prompt, and upstream does not define it.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Optional

import httpx
from PIL import Image

from urakatsu.core.config import GenerationConfig, ModelSelection
from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage
from urakatsu.runtimes.gguf_metadata import read_metadata, special_token_text
from urakatsu.runtimes.store import HuggingFaceFile, ModelStore

from . import prompt
from .backend import TranslationRequest
from .base import TranslationSettings, TranslatorProvider
from .catalog import DEFAULT_MODEL, DEFAULT_QUANTIZATION, MODELS, LocalModelDescriptor, ModelGeneration, QuantizationDefinition
from .errors import UnsupportedLanguage
from .model import Model, Quantization

# prompt plus generation must fit; one window for the loaded model.
N_CTX = 8192


@dataclass
class _Sampling:
    max_tokens: int
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    min_p: Optional[float]
    repeat_penalty: float
    frequency_penalty: float
    presence_penalty: float


def descriptor_for(model_id: Optional[str]) -> LocalModelDescriptor:
    """The built-in default, or a GGUF chat model installed from the Models page."""
    for descriptor in MODELS:
        if descriptor.id == model_id:
            return descriptor
    if model_id:
        from urakatsu.catalog.installed import InstalledModels

        installed = InstalledModels().get(model_id)
        if installed is not None and installed.runtime == "llama":
            return installed_descriptor(installed)
    raise ValueError(f"unknown local translator '{model_id}'; install it on the Models page")


def installed_descriptor(installed) -> LocalModelDescriptor:
    """An installed catalog model as a descriptor: one quantization, no curated sampling or language list."""
    model = next(f.path for f in installed.files if f.role == "model")
    projector = next((f.path for f in installed.files if f.role == "projector"), None)
    return LocalModelDescriptor(
        id=installed.id,
        name=f"{installed.repo.split('/')[-1]} {installed.variant}".strip(),
        reasoning=installed.reasoning,
        quantizations=(QuantizationDefinition(installed.variant or "default", installed.variant or "default", model),),
        generation=ModelGeneration(),
        repository=installed.repo,
        revision=installed.revision,
        projector=projector,
        target_languages=None,
    )


def sampling_options(descriptor: LocalModelDescriptor, overrides: GenerationConfig) -> _Sampling:
    """``ModelGeneration::options``: configuration overrides, then the catalog, then the defaults."""
    model: ModelGeneration = descriptor.generation

    def pick(name: str, default: Any = None) -> Any:
        value = getattr(overrides, name)
        if value is None:
            value = getattr(model, name)
        return default if value is None else value

    return _Sampling(
        max_tokens=int(pick("max_tokens", 1000)),
        temperature=float(pick("temperature", 0.1)),
        top_k=pick("top_k"),
        top_p=pick("top_p"),
        min_p=pick("min_p"),
        repeat_penalty=float(pick("repeat_penalty", 1.2)),
        frequency_penalty=float(pick("frequency_penalty", 0.0)),
        presence_penalty=float(pick("presence_penalty", 0.0)),
    )


def weight_sources(model_id: str, quantization: Optional[str]) -> tuple[HuggingFaceFile, ...]:
    descriptor = descriptor_for(model_id)
    definition = descriptor.quantization(quantization)
    sources = [HuggingFaceFile(descriptor.repository, definition.filename, descriptor.revision)]
    if descriptor.projector:
        sources.append(HuggingFaceFile(descriptor.repository, descriptor.projector, descriptor.revision))
    return tuple(sources)


@register
class LocalTranslator(TranslatorProvider):
    # N_CTX also holds the prompt, the page's segments, the page image and up to 1000 output tokens.
    context_budget: ClassVar[int] = 2000

    id: ClassVar[str] = "local"
    name: ClassVar[str] = "Local"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "On-device GGUF translation models (llama.cpp)"
    runtimes: ClassVar[tuple[str, ...]] = ("llama",)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """Any GGUF chat model (it ships a chat template); OCR-only architectures are left out."""
        if facts.runtime == "llama" and facts.chat_template and facts.architecture not in ("paddleocr",):
            return "llama"
        return None
    Settings: ClassVar[type[TranslationSettings]] = TranslationSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._server: Any = None
        self._template: str = ""
        self._bos = ""
        self._eos = ""

    # -- weights ---------------------------------------------------------------------------------

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        """Declared weights of the default model; the selected model's weights come from ``resolve_weights``."""
        return weight_sources(DEFAULT_MODEL, DEFAULT_QUANTIZATION)

    def resolve_weights(self, progress=None) -> list[Path]:
        return [self.resolve(source, progress) for source in self._sources()]

    def _selection(self) -> ModelSelection:
        return self.settings.translation.model

    def _sources(self) -> tuple[HuggingFaceFile, ...]:
        selection = self._selection()
        return weight_sources(selection.model or "", selection.quantization)

    # -- lifecycle -------------------------------------------------------------------------------

    def _load(self) -> None:
        descriptor = descriptor_for(self._selection().model)
        paths = self.resolve_weights()
        projector = paths[1] if descriptor.projector else None
        template = read_metadata(paths[0], {"tokenizer.chat_template"}).get("tokenizer.chat_template")
        if not template:
            raise ProcessingError("GGUF model does not contain a chat template")
        self._template = template
        self._bos = special_token_text(paths[0], "bos")
        self._eos = special_token_text(paths[0], "eos")
        self._server = self.runtime_instance().launch(paths[0], projector, context=N_CTX)

    def _unload(self) -> None:
        if self._server is not None:
            self._server.close()
        self._server = None

    # -- inference -------------------------------------------------------------------------------

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        """``local::models``: every catalog entry, with per-quantization download state."""
        store = ModelStore()
        models = []
        from urakatsu.catalog.installed import InstalledModels

        installed = [
            installed_descriptor(m)
            for m in InstalledModels(store).all()
            if m.runtime == "llama" and any(u.get("provider") == cls.id for u in m.usable_by)
        ]
        for descriptor in (*MODELS, *installed):
            quantizations = []
            for definition in descriptor.quantizations:
                files = [HuggingFaceFile(descriptor.repository, definition.filename, descriptor.revision)]
                if descriptor.projector:
                    files.append(HuggingFaceFile(descriptor.repository, descriptor.projector, descriptor.revision))
                quantizations.append(
                    Quantization(
                        id=definition.id,
                        name=definition.name,
                        downloaded=all(source.local_path(store).exists() for source in files),
                    )
                )
            models.append(
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=descriptor.id,
                    name=descriptor.name,
                    quantizations=quantizations,
                    vision=descriptor.projector is not None,
                    reasoning=descriptor.reasoning,
                )
            )
        return models

    def supports_vision_model(self, selection: ModelSelection) -> bool:
        """``local::supports_vision``: the catalog entry has a vision projector."""
        return selection.model is not None and descriptor_for(selection.model).projector is not None

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        descriptor = descriptor_for(self.model_id())
        if not descriptor.supports_language(request.target_language):
            raise UnsupportedLanguage(self.id, request.target_language.display_name)
        enable_thinking = bool(generation.reasoning) and descriptor.reasoning
        system, payload = prompt.prompts(request)
        schema = prompt.output_schema(len(request.segments))
        options = sampling_options(descriptor, generation)
        text = await asyncio.to_thread(
            self._infer, system, payload, request.image, enable_thinking, schema, options
        )
        return prompt.translations(self.id, text, request.segments)

    def _infer(
        self,
        system: str,
        payload: str,
        image: Optional[Image.Image],
        enable_thinking: bool,
        schema: dict,
        options: _Sampling,
    ) -> str:
        server = self._server
        if server is None:
            raise ProcessingError("local translator is not loaded")
        images: list[bytes] = []
        content = payload
        if image is not None:
            # The media marker goes before the payload, separated by a newline.
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            images.append(buffer.getvalue())
            content = f"{server.media_marker}\n{payload}"
        prompt_text = self._render([{"role": "system", "content": system}, {"role": "user", "content": content}], enable_thinking)
        return server.complete(
            server.strip_bos(prompt_text),
            images=images,
            json_schema=schema,
            n_predict=options.max_tokens,
            temperature=options.temperature,
            # Unset samplers are disabled explicitly; llama-server's own defaults would apply otherwise.
            top_k=options.top_k or 0,
            top_p=options.top_p or 1.0,
            min_p=options.min_p or 0.0,
            repeat_penalty=options.repeat_penalty,
            frequency_penalty=options.frequency_penalty,
            presence_penalty=options.presence_penalty,
        )

    def _render(self, messages: list[dict], enable_thinking: bool) -> str:
        """The GGUF chat template with ``add_generation_prompt`` and ``enable_thinking`` (``ChatTemplate::render``)."""
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        # minijinja's ``Environment::new()`` defaults: no trim_blocks / lstrip_blocks.
        environment = ImmutableSandboxedEnvironment(extensions=[loopcontrols])

        def raise_exception(message: str) -> None:
            raise ValueError(message)

        def strftime_now(format: str) -> str:
            return datetime.now().strftime(format)

        return environment.from_string(self._template).render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            bos_token=self._bos,
            eos_token=self._eos,
            tools=[],
            documents=[],
            raise_exception=raise_exception,
            strftime_now=strftime_now,
        )
