"""Translation model listing types.

``ModelSelection`` and ``GenerationConfig`` live in ``urakatsu.core.config`` (they
are part of the pipeline configuration).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Quantization(BaseModel):
    id: str
    name: str
    downloaded: bool


class Model(BaseModel):
    provider: str
    provider_name: str
    model: str | None
    name: str
    quantizations: list[Quantization] = Field(default_factory=list)
    vision: bool = False
    reasoning: bool = False


def service_model(provider: str, provider_name: str, name: str) -> Model:
    """``Model::service``: a hosted service with no model id and no quantizations."""
    return Model(provider=provider, provider_name=provider_name, model=None, name=name)


def display_name(model: str) -> str:
    """``display_name``: last path segment, split on ``-``/``_``, each word capitalised."""
    model = model.rsplit("/", 1)[-1]
    parts = [part for part in model.replace("_", "-").split("-") if part]
    return " ".join(part[:1].upper() + part[1:] for part in parts)
