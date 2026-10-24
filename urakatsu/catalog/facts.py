"""What the catalog knows about a Hugging Face repository, in the terms providers match on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .support import Family

if TYPE_CHECKING:  # pragma: no cover
    from .recipes import Recipe


@dataclass
class ModelFacts:
    runtime: str  # llama | diffusion | torch | onnx
    repo: str
    pipeline_tag: Optional[str] = None
    library: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    base_models: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    architecture: Optional[str] = None  # GGUF general.architecture
    chat_template: bool = False  # GGUF ships a chat template (a chat model)
    projector: bool = False  # GGUF repo ships an mmproj vision projector
    model_type: Optional[str] = None  # config.json model_type (Transformers)
    architectures: list[str] = field(default_factory=list)  # config.json architectures
    family: Optional[Family] = None  # stable-diffusion.cpp family
    recipe: Optional["Recipe"] = None  # stable-diffusion.cpp sd-server recipe
    # A diffusion-runtime repo that is a language model (its GGUF architecture is one llama.cpp
    # runs): a text encoder stable-diffusion.cpp reads prompts with (--llm), not an image model.
    text_encoder: bool = False

    def has_file(self, name: str) -> bool:
        return any(path == name or path.endswith("/" + name) for path in self.files)

    def names(self) -> list[str]:
        """Repo id and base models, lowercased, for name-based family checks."""
        return [self.repo.lower(), *(b.lower() for b in self.base_models)]
