"""stable-diffusion.cpp recipes: what any supported image model needs and how to run it.

``sd-server`` ships one JSON recipe per model in ``examples/server/recipes/`` (upstream
maintains them with the runtime, under its own policy): the component **slots** a model
needs (``diffusion_model``, ``llm``, ``vae``, ...), its default sampler settings, and the
features it supports (``ref_images`` for instruction editing; ``img2img`` and ``mask_image``
unless disabled). Nothing model-specific is written here:

- a Hugging Face repo is matched to a recipe by its ``model_url`` (the author's repo) in
  the repo id or base models, else by the recipe's name;
- a slot's file comes from the repo itself (ComfyUI-style ``text_encoders/``, ``vae/``
  folders) or from the companion links in the family's ``docs/<family>.md``
  ("Download weights"), both read at the installed release.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .support import Family, normalize

# Slots whose models add a conditioning input Urakatsu cannot supply (a control image, an
# identity photo, a motion module). A recipe that requires one cannot clean a page.
ADAPTER_SLOTS = frozenset({"control_net", "ip_adapter", "photo_maker", "pulid_weights", "motion_module", "audio_encoder", "audio_vae"})
# Where a ComfyUI-style repo keeps each slot's files (sd-server's own category folders).
SLOT_FOLDERS: dict[str, tuple[str, ...]] = {
    "model": ("checkpoints",),
    "diffusion_model": ("diffusion_models", "unet", "transformer"),
    "llm": ("text_encoders", "text_encoder"),
    "t5xxl": ("text_encoders", "text_encoder"),
    "clip_l": ("text_encoders", "text_encoder"),
    "clip_g": ("text_encoders", "text_encoder"),
    "llm_vision": ("text_encoders", "clip_vision"),
    "clip_vision": ("clip_vision",),
    "vae": ("vae",),
    "tae": ("vae", "vae_approx"),
    "tokenizer": ("tokenizers", "tokenizer"),
    "embeddings_connectors": ("text_encoders",),
}
WEIGHT_SUFFIXES = (".gguf", ".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin")
# Formats sd.cpp does not read (Apple MLX, GPTQ/AWQ packs) and files that are not whole models.
UNSUPPORTED = re.compile(r"(?i)(mlx|gptq|awq|w4a8|nvfp4|lora|-of-\d{5}|\.index\.json$)")
# For large companions (text encoders): smaller formats that keep quality come first.
QUANT_PREFERENCE = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "int8", "fp8", "q4_k_s", "q4_0", "q5_0", "fp16", "f16", "bf16", "fp32", "f32")


@dataclass(frozen=True)
class Slot:
    id: str
    required: bool
    label: str = ""


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    model_url: str = ""
    slots: tuple[Slot, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()
    defaults: tuple[tuple[str, Any], ...] = ()
    features: frozenset[str] = frozenset()
    disabled: frozenset[str] = frozenset()

    @property
    def model_repo(self) -> str:
        """``Qwen/Qwen-Image-2.1`` from ``https://huggingface.co/Qwen/Qwen-Image-2.1``."""
        match = re.match(r"https://huggingface\.co/([^/]+/[^/?#]+)", self.model_url or "")
        return match.group(1) if match else ""

    @property
    def main_slot(self) -> str:
        """The slot holding the model itself: a standalone ``diffusion_model`` or a full ``model`` checkpoint."""
        ids = [s.id for s in self.slots if s.required]
        return "diffusion_model" if "diffusion_model" in ids else "model"

    @property
    def edits(self) -> bool:
        """Native instruction / reference-image editing."""
        return "ref_images" in self.features

    def default(self, key: str, fallback: Any = None) -> Any:
        return dict(self.defaults).get(key, fallback)

    def needed(self) -> list[Slot]:
        """Required slots, plus the first slot of each ``required_any`` group (a full VAE before a TAE)."""
        chosen = [s for s in self.slots if s.required]
        for group in self.required_any:
            if not any(s.id in group for s in chosen):
                first = next((s for s in self.slots if s.id == group[0]), None)
                if first is not None:
                    chosen.append(first)
        return chosen

    def cleans_pages(self) -> bool:
        """Masked image-to-image works and no adapter input is required: it can repaint lettering."""
        if {"img2img", "mask_image"} & self.disabled:
            return False
        return not any(s.required and s.id in ADAPTER_SLOTS for s in self.slots)


def parse_recipe(data: dict[str, Any]) -> Recipe:
    return Recipe(
        id=str(data["id"]),
        name=str(data.get("name") or data["id"]),
        model_url=str(data.get("model_url") or ""),
        slots=tuple(Slot(str(s["id"]), bool(s.get("required")), str(s.get("label") or "")) for s in data.get("slots") or []),
        required_any=tuple(tuple(group) for group in data.get("required_any") or []),
        defaults=tuple(sorted((data.get("defaults") or {}).items())),
        features=frozenset(data.get("features") or []),
        disabled=frozenset(data.get("disabled_features") or []),
    )


def recipe_to_json(recipe: Recipe) -> dict[str, Any]:
    return {
        "id": recipe.id,
        "name": recipe.name,
        "model_url": recipe.model_url,
        "slots": [{"id": s.id, "required": s.required, "label": s.label} for s in recipe.slots],
        "required_any": [list(g) for g in recipe.required_any],
        "defaults": dict(recipe.defaults),
        "features": sorted(recipe.features),
        "disabled_features": sorted(recipe.disabled),
    }


def match_recipe(recipes: Iterable[Recipe], repo: str, base_models: Iterable[str] = ()) -> Optional[Recipe]:
    """The recipe for a Hugging Face repo: its author repo is the repo or a base model, else the
    longest recipe name or id found in the repo's name (repacks such as ``Comfy-Org/MOSS-Image``)."""
    recipes = [r for r in recipes]
    named = {repo.lower(), *(b.lower() for b in base_models)}
    # The repo itself first: a fine-tune's base model (SFT on Base) names a different recipe.
    for candidates in ({repo.lower()}, named):
        for recipe in recipes:
            if recipe.model_repo and recipe.model_repo.lower() in candidates:
                return recipe
    texts = [normalize(n.split("/")[-1]) for n in named]
    best: Optional[tuple[int, Recipe]] = None
    for recipe in recipes:
        for alias in {normalize(recipe.name), normalize(recipe.id), normalize(recipe.model_repo.split("/")[-1])}:
            if len(alias) >= 4 and any(alias in text for text in texts) and (best is None or len(alias) > best[0]):
                best = (len(alias), recipe)
    if best:
        return best[1]
    # A repack named after the family covers its variants ("MOSS-Image" holds Base and SFT,
    # which share their components): the first recipe whose name starts with the repo's name.
    for recipe in sorted(recipes, key=lambda r: r.id):
        if any(len(text) >= 6 and normalize(recipe.name).startswith(text) for text in texts):
            return recipe
    return None


def family_for(families: Iterable[Family], recipe: Recipe) -> Optional[Family]:
    """The README family (and so the ``docs/`` page) documenting a recipe."""
    target = {normalize(recipe.name), normalize(recipe.id)}
    best: Optional[tuple[int, Family]] = None
    for family in families:
        for alias in family.aliases:
            if any(alias == t or (len(alias) >= 5 and alias in t) for t in target) and (best is None or len(alias) > best[0]):
                best = (len(alias), family)
    return best[1] if best else None


# ---- companion sources from a family doc ---------------------------------------------------------


@dataclass(frozen=True)
class Source:
    repo: str
    folder: str = ""  # a subfolder (``tree/main/<folder>``) or a single file (``blob/main/<file>``)
    file: str = ""


_LINK = re.compile(r"https://huggingface\.co/([\w.-]+/[\w.-]+)(?:/(tree|blob|resolve)/[^/\s)]+/([^\s)`]+))?")


def parse_sources(doc: str, recipe: Recipe) -> dict[str, list[Source]]:
    """Slot -> companion sources from the doc's "Download weights" bullets.

    The first bullet is the model itself; a bullet naming a slot (``vae``, ``clip_l``,
    ``t5xxl``, ``tokenizer``, ``tae``) is that slot; the remaining one is the text encoder
    (``llm``) when the recipe has one. A projector mentioned with a text encoder's GGUF
    repository is taken from that repository."""
    section = re.split(r"(?im)^#+\s*download", doc, maxsplit=1)
    if len(section) < 2:
        return {}
    body = re.split(r"(?m)^#+\s", section[1], maxsplit=1)[0]
    bullets = re.split(r"(?m)^- ", body)[1:]
    slots = {s.id for s in recipe.slots}
    found: dict[str, list[Source]] = {}
    for index, bullet in enumerate(bullets):
        title = bullet.split("\n", 1)[0].lower()
        sources = [Source(repo, file=path) if kind in ("blob", "resolve") else Source(repo, folder=path or "") for repo, kind, path in _LINK.findall(bullet)]
        if not sources:
            continue
        slot = _slot_for(title, index, slots)
        if slot:
            found.setdefault(slot, []).extend(sources)
    return found


def _slot_for(title: str, index: int, slots: set[str]) -> Optional[str]:
    for word, slot in (("tokenizer", "tokenizer"), ("taesd", "tae"), ("tae", "tae"), ("vae", "vae"), ("clip_l", "clip_l"), ("clip-l", "clip_l"), ("clip_g", "clip_g"), ("clip-g", "clip_g"), ("t5", "t5xxl"), ("clip_vision", "clip_vision"), ("clip vision", "clip_vision")):
        if word in title and slot in slots:
            return slot
    if index == 0:
        return "diffusion_model" if "diffusion_model" in slots else "model"
    return "llm" if "llm" in slots else None


# ---- picking files -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    repo: str
    path: str
    size: int
    revision: str = ""


def slot_candidates(slot: str, repo: str, files: list[tuple[str, int]], within: str = "", anywhere: bool = False, revision: str = "") -> list[Candidate]:
    """Files in ``files`` that can fill ``slot``: in the slot's folder, or named after it.

    ``within`` limits the search to a folder or file; ``anywhere`` accepts every weight file
    (a repository the doc links for exactly this slot, such as a text encoder's GGUF repo)."""
    out = []
    for path, size in files:
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        if within and not lower.startswith(within.lower().rstrip("/") + "/") and lower != within.lower():
            continue
        if UNSUPPORTED.search(lower):
            continue
        if slot == "tokenizer":
            if name == "tokenizer.json":
                out.append(Candidate(repo, path, size, revision))
            continue
        if not name.endswith(WEIGHT_SUFFIXES):
            continue
        is_projector = name.startswith("mmproj") or "mmproj" in name
        if slot == "llm_vision":
            if is_projector:
                out.append(Candidate(repo, path, size, revision))
            continue
        if is_projector:
            continue
        folders = SLOT_FOLDERS.get(slot, ())
        in_folder = any(f"{folder}/" in lower for folder in folders)
        named = slot in name.replace("-", "_") or (slot == "vae" and "vae" in name) or (slot == "t5xxl" and "t5" in name)
        if within or anywhere or in_folder or named:
            if slot == "vae" and not ("vae" in lower or within or anywhere or in_folder):
                continue
            if slot in ("llm", "t5xxl", "clip_l", "clip_g") and "vae" in name:
                continue
            out.append(Candidate(repo, path, size, revision))
    return out


def pick(slot: str, candidates: list[Candidate]) -> Optional[Candidate]:
    """One file per slot: text encoders in the smallest format that keeps quality, a full VAE
    rather than a small or tiny decoder, the projector closest to full precision."""
    if not candidates:
        return None
    if slot == "vae":
        full = [c for c in candidates if not re.search(r"(?i)small|tiny|tae|preview", c.path)]
        return max(full or candidates, key=lambda c: c.size)
    if slot == "llm_vision":
        for wanted in ("f16", "bf16", "q8_0", "f32"):
            hit = next((c for c in candidates if re.search(rf"(?i)[-_.]{wanted}\.gguf$", c.path)), None)
            if hit:
                return hit
        return min(candidates, key=lambda c: c.size)
    if slot == "tokenizer":
        return candidates[0]

    def rank(c: Candidate) -> tuple[int, int]:
        name = c.path.lower()
        for i, quant in enumerate(QUANT_PREFERENCE):
            if re.search(rf"(?:^|[-_.]){re.escape(quant)}(?:[-_.]|$)", name.rsplit("/", 1)[-1]):
                return (i, c.size)
        return (len(QUANT_PREFERENCE), c.size)

    return min(candidates, key=rank)


@dataclass
class Plan:
    """The files a model needs: the chosen main weights plus one file per needed slot."""

    recipe: Recipe
    files: dict[str, Candidate] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.files.values())
