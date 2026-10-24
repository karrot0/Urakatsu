"""The model catalog: Hugging Face search, per-variant hardware fit, runtime compatibility, installs.

For every repository the catalog answers three questions:

1. Which stages can use it? Each provider's ``accepts(facts)`` decides (no model lists).
2. Will it run on this machine, and which quantization? (``fit``)
3. Can the installed runtime version load it? (``support``) A pinned older runtime gets a
   warning instead of a failure; the newest release is checked to say whether updating helps.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from urakatsu.core.config import home
from urakatsu.core.registry import registry
from urakatsu.runtimes import models_folder
from urakatsu.runtimes.hardware import discover, system_profile
from urakatsu.runtimes.packages import LLAMA_CPP, PACKAGES, STABLE_DIFFUSION_CPP, PackageManager
from urakatsu.runtimes.store import ModelStore

from . import fit as fitting
from .diskcache import DiskCache
from .facts import ModelFacts
from .hf import HubClient, RepoInfo
from .installed import Downloader, InstalledFile, InstalledModel, InstalledModels
from .recipes import UNSUPPORTED, Plan, family_for, match_recipe, parse_sources, pick, slot_candidates
from .support import SupportIndex, match_family

RUNTIME_PACKAGE = {"llama": LLAMA_CPP, "diffusion": STABLE_DIFFUSION_CPP}
RUNTIME_LABEL = {"llama": "llama.cpp", "diffusion": "stable-diffusion.cpp", "torch": "PyTorch", "onnx": "ONNX Runtime"}
CONTEXT_FOR_STAGE = {"translation": 8192, "ocr": 4096}
TORCH_FILES = re.compile(r"(?i)(\.json|\.safetensors|\.txt|\.model|tokenizer[^/]*|vocab[^/]*|merges[^/]*)$")
ONNX_FILES = re.compile(r"(?i)(\.onnx|\.onnx_data|\.json|\.txt)$")
PROJECTOR_PREFERENCE = ("F16", "BF16", "Q8_0", "F32")
# Files a diffusion repo ships beside the model itself (ComfyUI and diffusers folders, and
# companions named after their role); the model's own weights are everything else.
COMPANION_FOLDER = re.compile(r"(?i)(^|/)(text_encoders?|vae|vae_approx|clip_vision|loras?|tokenizers?|scheduler|controlnet|unet|transformer|feature_extractor|image_encoder)/")
COMPANION = re.compile(r"(?i)(text_encoder|vae|clip|t5xxl|umt5|qwen_?\d|llm|tokenizer)")
SLOT_LABELS = {
    "llm": "Text encoder",
    "llm_vision": "Vision projector",
    "vae": "VAE",
    "tae": "Tiny VAE",
    "t5xxl": "T5 text encoder",
    "clip_l": "CLIP-L text encoder",
    "clip_g": "CLIP-G text encoder",
    "tokenizer": "Tokenizer",
    "clip_vision": "CLIP vision encoder",
    "diffusion_model": "Diffusion model",
    "model": "Checkpoint",
}
LATEST_TTL = 1800  # seconds a newest-release lookup is reused
PYTHON_RUNTIMES = (("torch", "PyTorch"), ("torchvision", "torchvision"), ("transformers", "Transformers"), ("onnxruntime-gpu", "ONNX Runtime GPU"), ("onnxruntime", "ONNX Runtime"))


class Catalog:
    def __init__(self, store: Optional[ModelStore] = None, hub: Optional[HubClient] = None):
        self.store = store or ModelStore()
        self.hub = hub or HubClient(cache_dir=home() / "cache" / "hub")
        self.packages = PackageManager()
        self.support = SupportIndex(self.packages)
        self.installed = InstalledModels(self.store)
        self.downloads = Downloader(self.installed)
        self._latest: dict[str, tuple[str, float]] = {}
        self._latest_disk = DiskCache(home() / "cache" / "runtimes") if hub is None else None
        self._latest_refreshing: set[str] = set()
        registry.discover()

    # -- hardware ----------------------------------------------------------------------------------

    def hardware(self) -> dict[str, Any]:
        devices = discover().candidates()
        system = system_profile()
        return {
            "devices": [d.model_dump(mode="json") for d in devices if d.is_accelerator],
            "system": system.model_dump(mode="json"),
            "usable_vram": fitting.usable_vram(devices, system),
        }

    # -- facts and usage ---------------------------------------------------------------------------

    def facts(self, runtime: str, repo: RepoInfo) -> ModelFacts:
        files = [f.path for f in repo.files]
        family = None
        recipe = None
        text_encoder = False
        if runtime == "diffusion":
            latest = self._latest_support(runtime)
            if latest is not None:
                family = match_family(latest.families, [repo.id, *repo.base_models])
                recipe = match_recipe(latest.recipes, repo.id, repo.base_models)
                if family is None and recipe is not None:
                    family = family_for(latest.families, recipe)
            # Names can mislead: "Qwen-Image-2.1-...-GGUF" may hold only the Qwen3-VL text encoder.
            # Diffusion GGUFs declare their own architectures (qwen_image21, flux); one llama.cpp
            # runs is a language model, which can never fill a recipe's image-model slot.
            text_encoder = self.is_language_model(repo.gguf.get("architecture"))
            if text_encoder:
                recipe = None
        return ModelFacts(
            runtime=runtime,
            repo=repo.id,
            pipeline_tag=repo.pipeline_tag,
            library=repo.library,
            tags=repo.tags,
            base_models=repo.base_models,
            files=files,
            architecture=repo.gguf.get("architecture"),
            chat_template=bool(repo.gguf.get("chat_template")),
            projector=any(fitting.is_projector(f) for f in files) or repo.gguf.get("architecture") == "clip",
            model_type=(repo.config or {}).get("model_type"),
            architectures=list((repo.config or {}).get("architectures") or []),
            family=family,
            recipe=recipe,
            text_encoder=text_encoder,
        )

    def is_language_model(self, architecture: Optional[str]) -> bool:
        """True for a GGUF architecture llama.cpp runs (qwen3vl, t5encoder...), from its own sources."""
        if not architecture:
            return False
        support = self._latest_support("llama") or self.support.installed(LLAMA_CPP)
        return support is not None and architecture in support.architectures

    def usable_by(self, facts: ModelFacts) -> list[dict[str, str]]:
        if facts.runtime == "torch" and not _has_safetensors(facts):
            return []  # pickle checkpoints (.bin / .pt) can run code on load; only safetensors are installed
        usage = []
        for stage, providers in registry._providers.items():
            for provider_id, cls in providers.items():
                runtime = cls.accepts(facts)
                if runtime:
                    usage.append({"stage": stage.value, "provider": provider_id, "name": cls.name or provider_id, "runtime": runtime})
        return usage

    # -- runtime compatibility ---------------------------------------------------------------------

    def latest_tag(self, package) -> Optional[str]:
        """The newest release tag, cached for a while so searches do not hit GitHub every time.

        Saved on disk too: after a restart the saved tag is used at once (refreshed in the
        background once older than LATEST_TTL) instead of a GitHub round trip per model.
        """
        cached = self._latest.get(package.id)
        if cached is None and self._latest_disk is not None:
            saved = self._latest_disk.get(f"latest:{package.id}")
            if saved and isinstance(saved[1], str):
                cached = self._latest[package.id] = (saved[1], time.time() - saved[0])
        if cached and time.time() - cached[1] < LATEST_TTL:
            return cached[0]
        if self.packages.offline:
            return cached[0] if cached else None
        if cached:
            self._refresh_latest(package)
            return cached[0]
        return self._fetch_latest(package)

    def _fetch_latest(self, package) -> Optional[str]:
        try:
            tag = self.packages.latest(package, self._variants()).tag
        except Exception:  # noqa: BLE001 - offline or GitHub unreachable
            cached = self._latest.get(package.id)
            return cached[0] if cached else None
        self._latest[package.id] = (tag, time.time())
        if self._latest_disk is not None:
            self._latest_disk.put(f"latest:{package.id}", tag)
        return tag

    def _refresh_latest(self, package) -> None:
        if package.id in self._latest_refreshing:
            return
        self._latest_refreshing.add(package.id)

        def run() -> None:
            try:
                self._fetch_latest(package)
            finally:
                self._latest_refreshing.discard(package.id)

        threading.Thread(target=run, name=f"latest-{package.id}", daemon=True).start()

    def _latest_support(self, runtime: str):
        package = RUNTIME_PACKAGE.get(runtime)
        if package is None:
            return None
        tag = self.latest_tag(package)
        if tag is None:
            return self.support.installed(package)
        return self.support.for_tag(package, tag)

    def _variants(self) -> list[str]:
        from urakatsu.runtimes.packages import preferred_variants

        return preferred_variants(discover().candidates())

    def compatibility(self, facts: ModelFacts) -> dict[str, Any]:
        """``supported`` | ``needs_update`` | ``unsupported`` | ``not_installed`` | ``unknown``, with a message."""
        package = RUNTIME_PACKAGE.get(facts.runtime)
        if facts.text_encoder and package is not None:
            family = f"{facts.family.name} " if facts.family else ""
            return {
                "status": "unsupported",
                "runtime": package.name,
                "installed": None,
                "latest": None,
                "pinned": None,
                "message": (
                    f"This is a text encoder (a {facts.architecture} language model), not an image model. "
                    f"stable-diffusion.cpp reads prompts with it next to a {family}denoiser; "
                    "install the denoiser to clean pages, and it brings its own text encoder."
                ),
            }
        if package is None:
            return self._python_compatibility(facts)
        installed = self.packages.installed(package)
        current = self.support.installed(package) if installed else None
        latest = self._latest_support(facts.runtime)

        def supports(support) -> Optional[bool]:
            if support is None:
                return None
            if facts.runtime == "llama":
                return facts.architecture in support.architectures if facts.architecture else None
            return facts.family is not None and support.family(facts.family.doc) is not None

        now, newest = supports(current), supports(latest)
        thing = f"architecture “{facts.architecture}”" if facts.runtime == "llama" else (f"the {facts.family.name} family" if facts.family else "this model family")
        result = {"runtime": package.name, "installed": installed.tag if installed else None, "latest": latest.tag if latest else None, "pinned": self.packages.pin(package)}
        if installed is None:
            if newest is False:
                return {**result, "status": "unsupported", "message": f"No {package.name} release supports {thing} yet."}
            supported = f" {latest.tag} supports {thing}." if newest else ""
            return {**result, "status": "not_installed", "message": f"{package.name} installs the first time a model needs it.{supported}"}
        if now:
            return {**result, "status": "supported", "message": f"{package.name} {installed.tag} supports {thing}."}
        if now is False and newest:
            hint = " Unpin it in Settings, Runtimes to update." if result["pinned"] else " Update it in Settings, Runtimes."
            return {**result, "status": "needs_update", "message": f"{package.name} {installed.tag} cannot load {thing}; {latest.tag} can.{hint}"}
        if now is False or newest is False:
            return {**result, "status": "unsupported", "message": f"No {package.name} release supports {thing} yet."}
        return {**result, "status": "unknown", "message": f"Could not check {package.name} support for this model."}

    def _python_compatibility(self, facts: ModelFacts) -> dict[str, Any]:
        if facts.runtime == "torch" and facts.files and not _has_safetensors(facts):
            return {"status": "unsupported", "runtime": "PyTorch", "message": "This repository only ships pickle weights (.bin or .pt). Urakatsu loads .safetensors only."}
        if facts.runtime == "torch" and facts.model_type:
            try:
                import importlib.metadata as metadata

                from transformers import CONFIG_MAPPING

                version = metadata.version("transformers")
                if facts.model_type in CONFIG_MAPPING:
                    return {"status": "supported", "runtime": "Transformers", "installed": version, "message": f"Transformers {version} knows model type “{facts.model_type}”."}
                return {"status": "supported", "runtime": "Transformers", "installed": version, "message": "Loaded by Urakatsu's own model code."}
            except ImportError:
                return {"status": "unsupported", "runtime": "PyTorch", "message": "PyTorch support is not installed (pip install 'urakatsu[torch]')."}
        return {"status": "unknown", "runtime": facts.runtime, "message": ""}

    # -- search ------------------------------------------------------------------------------------

    def search(self, runtime: str, query: str = "", sort: str = "downloads", limit: int = 30, usable_only: bool = False) -> list[dict[str, Any]]:
        hardware = discover().candidates()
        system = system_profile()
        installed_repos = {m.repo for m in self.installed.all()}
        results = []
        repos = self.hub.search(runtime, query, sort=sort, limit=limit)
        if usable_only and not query:
            # The most downloaded repos of a runtime are mostly ones no stage can load; add what
            # each provider's own searches find so the list starts with usable models.
            seen = {r.id for r in repos}
            hints = dict.fromkeys(q for cls in registry.list() if runtime in cls.runtimes_declared() for q in cls.catalog_queries)
            for hint in hints:
                for repo in self.hub.search(runtime, hint, sort=sort, limit=10):
                    if repo.id not in seen:
                        seen.add(repo.id)
                        repos.append(repo)
        for repo in repos:
            facts = self.facts(runtime, repo)
            usage = self.usable_by(facts)
            if usable_only and not usage:
                continue
            estimate = None
            params = repo.gguf.get("total")
            if runtime == "llama" and params:
                # Before opening a repo: a typical 4-bit quantization is about 0.6 bytes per parameter.
                estimate = asdict(fitting.fit_llm(int(params * 0.6), 0, hardware, system))
            results.append(
                {
                    "repo": repo.id,
                    "author": repo.author,
                    "downloads": repo.downloads,
                    "likes": repo.likes,
                    "updated": repo.last_modified,
                    "pipeline_tag": repo.pipeline_tag,
                    "architecture": facts.architecture,
                    "model_type": facts.model_type,
                    "family": facts.family.name if facts.family else None,
                    "parameters": params,
                    "gated": repo.gated,
                    "usable_by": usage,
                    "estimate": estimate,
                    "installed": repo.id in installed_repos,
                }
            )
        if sort in ("downloads", "likes"):
            results.sort(key=lambda r: r[sort], reverse=True)
        return results

    # -- details -----------------------------------------------------------------------------------

    def details(self, runtime: str, repo_id: str) -> dict[str, Any]:
        repo = self.hub.model(repo_id)
        facts = self.facts(runtime, repo)
        devices = discover().candidates()
        system = system_profile()
        usage = self.usable_by(facts)
        plan = self.components(repo, facts) if runtime == "diffusion" else None
        variants = self._variants_for(runtime, repo, facts, devices, system, usage, plan)
        sizes = {v["id"]: v["size"] for v in variants}
        fits = {v["id"]: fitting.Fit(**v["fit"]) for v in variants if v.get("fit")}
        recommended = fitting.best_quantization(fits, sizes)
        for variant in variants:
            variant["recommended"] = variant["id"] == recommended
        installed = [m.model_dump(mode="json") for m in self.installed.all() if m.repo == repo.id]
        return {
            "repo": repo.id,
            "revision": repo.sha,
            "author": repo.author,
            "downloads": repo.downloads,
            "likes": repo.likes,
            "updated": repo.last_modified,
            "pipeline_tag": repo.pipeline_tag,
            "library": repo.library,
            "tags": [t for t in repo.tags if ":" not in t][:20],
            "gated": repo.gated,
            "architecture": facts.architecture,
            "model_type": facts.model_type,
            "family": facts.family.name if facts.family else None,
            "parameters": repo.gguf.get("total"),
            "context_length": repo.gguf.get("context_length"),
            "vision": facts.projector,
            "reasoning": "enable_thinking" in (repo.gguf.get("chat_template") or ""),
            "usable_by": usage,
            "compatibility": self.compatibility(facts),
            "variants": variants,
            "recommended": recommended,
            "installed": installed,
            "readme": _trim_readme(self.hub.readme(repo.id, repo.sha)),
            "hardware": self.hardware(),
            **self._recipe_details(repo, facts, plan, variants),
        }

    def _recipe_details(self, repo: RepoInfo, facts: ModelFacts, plan: Optional[Plan], variants: list[dict[str, Any]]) -> dict[str, Any]:
        """What a diffusion model needs beside its own weights, and where to find a usable copy."""
        if plan is None:
            return {"components": [], "missing": [], "alternatives": [], "recipe": None, "edits": False}
        labels = {s.id: s.label for s in plan.recipe.slots}
        components = [
            {"slot": slot, "label": labels.get(slot) or SLOT_LABELS.get(slot, slot), "path": c.path, "repo": c.repo, "size": c.size, "here": c.repo == repo.id}
            for slot, c in plan.files.items()
        ]
        alternatives: list[str] = []
        if not variants or plan.missing:
            sources = self._doc_sources(facts).get(plan.recipe.main_slot, [])
            alternatives = list(dict.fromkeys(s.repo for s in sources if s.repo != repo.id))
        return {
            "components": components,
            "missing": [SLOT_LABELS.get(m, m) for m in plan.missing],
            "alternatives": alternatives,
            "recipe": plan.recipe.name,
            "edits": plan.recipe.edits,
        }

    def _doc_sources(self, facts: ModelFacts) -> dict[str, list]:
        if facts.recipe is None or facts.family is None:
            return {}
        installed = self.packages.installed(STABLE_DIFFUSION_CPP)
        tag = self.latest_tag(STABLE_DIFFUSION_CPP) or (installed.tag if installed else None)
        if tag is None:
            return {}
        return parse_sources(self.support.doc(STABLE_DIFFUSION_CPP, tag, facts.family.doc), facts.recipe)

    def components(self, repo: RepoInfo, facts: ModelFacts) -> Optional[Plan]:
        """One file for every slot the model's recipe needs besides the model itself: from the
        repo first (ComfyUI repacks bundle them), else from the companions its doc links."""
        recipe = facts.recipe
        if recipe is None:
            return None
        plan = Plan(recipe)
        files = [(f.path, f.size) for f in repo.files]
        sources: Optional[dict[str, list]] = None
        for slot in recipe.needed():
            if slot.id == recipe.main_slot:
                continue
            chosen = pick(slot.id, slot_candidates(slot.id, repo.id, files, revision=repo.sha))
            if chosen is None:
                sources = self._doc_sources(facts) if sources is None else sources
                for source in sources.get(slot.id, []):
                    try:
                        info = self.hub.model(source.repo)
                    except Exception:  # noqa: BLE001 - try the next source
                        continue
                    found = slot_candidates(slot.id, info.id, [(f.path, f.size) for f in info.files], within=source.folder or source.file, anywhere=True, revision=info.sha)
                    chosen = pick(slot.id, found)
                    if chosen is not None:
                        break
            if chosen is not None:
                plan.files[slot.id] = chosen
            else:
                plan.missing.append(slot.id)
        # Editing needs the text encoder's vision weights: bundled in a full checkpoint, a
        # separate projector beside a GGUF one.
        llm = plan.files.get("llm")
        if recipe.edits and llm is not None and llm.path.lower().endswith(".gguf") and any(s.id == "llm_vision" for s in recipe.slots):
            try:
                info = repo if llm.repo == repo.id else self.hub.model(llm.repo)
                vision = pick("llm_vision", slot_candidates("llm_vision", info.id, [(f.path, f.size) for f in info.files], revision=info.sha))
            except Exception:  # noqa: BLE001 - edits then fall back to masked repainting
                vision = None
            if vision is not None:
                plan.files["llm_vision"] = vision
        return plan

    def _variants_for(self, runtime, repo: RepoInfo, facts: ModelFacts, devices, system, usage, plan: Optional[Plan] = None) -> list[dict[str, Any]]:
        files = [(f.path, f.size) for f in repo.files]
        if runtime == "diffusion":
            return self._diffusion_variants(files, plan, devices, system)
        if runtime == "llama" and any(p.lower().endswith(".gguf") for p, _ in files):
            groups = fitting.group_quantizations(files)
            projector = self._projector(files) if runtime == "llama" else None
            kv = 0
            if runtime == "llama" and groups and facts.architecture:
                context = max([CONTEXT_FOR_STAGE.get(u["stage"], 4096) for u in usage] or [4096])
                smallest = min(groups, key=lambda q: sum(s for _, s in groups[q]))
                try:
                    header = self.hub.gguf_header(repo.id, repo.sha, fitting.first_shard(groups[smallest]), fitting.header_keys(facts.architecture))
                    kv = fitting.kv_cache_bytes(header, context)
                except Exception:  # noqa: BLE001 - estimate without the KV cache
                    kv = 0
            variants = []
            for label, items in sorted(groups.items(), key=lambda kv_: sum(s for _, s in kv_[1])):
                size = sum(s for _, s in items) + (projector[1] if projector else 0)
                if runtime == "llama":
                    fit = fitting.fit_llm(size, kv, devices, system)
                else:
                    fit = fitting.fit_diffusion([size], devices, system)
                variants.append({"id": label, "label": label, "files": [p for p, _ in items] + ([projector[0]] if projector else []), "size": size, "fit": asdict(fit)})
            return variants
        pattern = TORCH_FILES if runtime == "torch" else ONNX_FILES
        chosen = [(p, s) for p, s in files if pattern.search(p) and not p.lower().endswith(("readme.md",))]
        if runtime == "torch" and any(p.endswith(".safetensors") for p, _ in chosen):
            chosen = [(p, s) for p, s in chosen if not p.endswith((".bin", ".pt", ".ckpt"))]
        size = sum(s for _, s in chosen)
        return [{"id": "snapshot", "label": "Full model", "files": [p for p, _ in chosen], "size": size, "fit": asdict(fitting.fit_dense(size, devices, system))}] if chosen else []

    @staticmethod
    def _diffusion_variants(files: list[tuple[str, int]], plan: Optional[Plan], devices, system) -> list[dict[str, Any]]:
        """The model's own weights (one per quantization), not the companions beside them.
        Each is rated with its companions: what the GPU and RAM must hold together."""
        companions = {c.path for c in plan.files.values()} if plan else set()
        own = [
            (p, s)
            for p, s in files
            if p not in companions
            and not COMPANION_FOLDER.search(p)
            and not UNSUPPORTED.search(p)
            and not fitting.is_projector(p)
            and p.lower().endswith((".gguf", ".safetensors", ".sft", ".ckpt"))
        ]
        named = [(p, s) for p, s in own if not COMPANION.search(p.rsplit("/", 1)[-1])]
        extra = [c.size for c in plan.files.values()] if plan else []
        variants = []
        ggufs = [(p, s) for p, s in named if p.lower().endswith(".gguf")]
        if ggufs:
            for label, items in sorted(fitting.group_quantizations(ggufs).items(), key=lambda kv: sum(s for _, s in kv[1])):
                size = sum(s for _, s in items)
                variants.append({"id": label, "label": label, "files": [p for p, _ in items], "size": size, "fit": asdict(fitting.fit_diffusion([size, *extra], devices, system))})
            return variants
        in_folder = [(p, s) for p, s in own if re.search(r"(?i)(^|/)(diffusion_models|checkpoints)/", p)]
        for path, size in sorted(in_folder or named, key=lambda x: x[1]):
            label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            variants.append({"id": path, "label": label, "files": [path], "size": size, "fit": asdict(fitting.fit_diffusion([size, *extra], devices, system))})
        return variants

    @staticmethod
    def _projector(files: list[tuple[str, int]]) -> Optional[tuple[str, int]]:
        projectors = [(p, s) for p, s in files if fitting.is_projector(p) and p.lower().endswith(".gguf")]
        if not projectors:
            return None
        for preferred in PROJECTOR_PREFERENCE:
            for path, size in projectors:
                if (fitting.quantization_of(path) or "") == preferred:
                    return path, size
        return min(projectors, key=lambda x: x[1])

    # -- install -----------------------------------------------------------------------------------

    def install(self, runtime: str, repo_id: str, variant: str) -> dict[str, Any]:
        if models_folder.mover.running:
            raise ValueError("Models are moving to a new folder. Install once the move finishes.")
        details_repo = self.hub.model(repo_id)
        facts = self.facts(runtime, details_repo)
        usage = self.usable_by(facts)
        devices, system = discover().candidates(), system_profile()
        plan = self.components(details_repo, facts) if runtime == "diffusion" else None
        if plan is not None and plan.missing:
            raise ValueError(f"{repo_id} is missing {', '.join(SLOT_LABELS.get(m, m) for m in plan.missing)} and no companion was found")
        chosen = next((v for v in self._variants_for(runtime, details_repo, facts, devices, system, usage, plan) if v["id"] == variant), None)
        if chosen is None:
            raise ValueError(f"{repo_id} has no variant {variant!r}")
        sizes = {f.path: f.size for f in details_repo.files}
        files = []
        main_done = False
        for path in chosen["files"]:
            if runtime == "llama":
                role = "projector" if fitting.is_projector(path) else ("model" if not main_done and path == fitting.first_shard([(p, 0) for p in chosen["files"] if not fitting.is_projector(p)]) else "shard")
                main_done = main_done or role == "model"
            elif runtime == "diffusion":
                role = plan.recipe.main_slot if plan else "diffusion_model"
            else:
                role = "file"
            files.append(InstalledFile(path=path, role=role, size=sizes.get(path, 0)))
        for slot, candidate in plan.files.items() if plan else ():
            elsewhere = candidate.repo != repo_id
            files.append(
                InstalledFile(
                    path=candidate.path,
                    role=slot,
                    size=candidate.size,
                    repo=candidate.repo if elsewhere else None,
                    revision=candidate.revision if elsewhere else None,
                )
            )
        model = InstalledModel(
            id=f"{repo_id}:{variant}" if variant != "snapshot" else repo_id,
            repo=repo_id,
            revision=details_repo.sha,
            runtime=runtime,
            variant="" if variant == "snapshot" else variant,
            files=files,
            architecture=facts.architecture,
            family=facts.family.doc if facts.family else None,
            model_type=facts.model_type,
            pipeline_tag=facts.pipeline_tag,
            chat_template=facts.chat_template,
            projector=any(f.role == "projector" for f in files),
            reasoning="enable_thinking" in (details_repo.gguf.get("chat_template") or ""),
            size=sum(f.size for f in files),
            usable_by=usage,
            recipe=plan.recipe.id if plan else None,
        )
        download = self.downloads.start(model)
        return download.model_dump(mode="json")

    def list_installed(self) -> list[dict[str, Any]]:
        """Installed models, each re-checked against the runtime version installed now."""
        result = []
        for model in self.installed.all():
            facts = ModelFacts(runtime=model.runtime, repo=model.repo, architecture=model.architecture, model_type=model.model_type, files=[f.path for f in model.files])
            if model.family:
                support = self.support.installed(STABLE_DIFFUSION_CPP) or self._latest_support("diffusion")
                facts.family = support.family(model.family) if support else None
                if facts.family is None and support is not None:
                    from .support import Family

                    facts.family = Family(model.family, model.family, ())
            entry = model.model_dump(mode="json")
            entry["compatibility"] = self.compatibility(facts) if model.runtime in RUNTIME_PACKAGE else {"status": "supported", "message": ""}
            entry["complete"] = self.installed.complete(model)
            # Editing models follow instructions ("replace X with Y"), not descriptions (the fill prompt says so).
            entry["edits"] = False
            if model.recipe:
                support = self.support.installed(STABLE_DIFFUSION_CPP) or self._latest_support("diffusion")
                recipe = support.recipe(model.recipe) if support else None
                entry["edits"] = bool(recipe and recipe.edits)
            result.append(entry)
        return sorted(result, key=lambda m: m["installed_at"], reverse=True)

    def delete(self, model_id: str) -> None:
        if models_folder.mover.running:
            raise ValueError("Models are moving to a new folder. Delete once the move finishes.")
        self.installed.remove(model_id)

    def list_downloads(self) -> list[dict[str, Any]]:
        cutoff = time.time() - 3600
        return [d.model_dump(mode="json") for d in self.downloads.downloads.values() if d.status in ("queued", "downloading") or d.finished_at > cutoff]

    # -- storage -----------------------------------------------------------------------------------

    def storage(self) -> dict[str, Any]:
        """The models folder, what it holds, its drive's space, and a move in progress."""
        root = self.store.root
        disk = shutil.disk_usage(models_folder.existing_ancestor(root))
        move = models_folder.mover.move
        items = self._storage_items(root)
        return {
            "models_dir": str(root),
            "default_dir": str(models_folder.default_root()),
            "size": sum(item["size"] for item in items),
            "items": items,
            "disk": {"total": disk.total, "used": disk.total - disk.free, "free": disk.free},
            "move": move.model_dump(mode="json") if move else None,
        }

    def _storage_items(self, root: Path) -> list[dict[str, Any]]:
        """What takes the space, largest first: installed models, the providers' built-in weights
        (named after the providers that use them), unfinished downloads, and anything else."""
        files: dict[str, int] = {}
        if root.exists():
            for path in root.rglob("*"):
                try:
                    if path.is_file():
                        files[path.relative_to(root).as_posix()] = path.stat().st_size
                except OSError:
                    pass
        items: list[dict[str, Any]] = []
        claimed: set[str] = set()
        for model in self.installed.all():
            size = 0
            for file in model.files:
                try:
                    rel = model.source(file.path).local_path(self.store).relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel in files and rel not in claimed:  # a file two variants share counts once
                    claimed.add(rel)
                    size += files[rel]
            items.append({"id": model.id, "kind": "installed", "name": model.repo, "detail": model.variant or RUNTIME_LABEL.get(model.runtime, model.runtime), "size": size})

        users: dict[str, list[str]] = {}
        for provider in registry.info():
            for weight in provider.weights:
                scheme, _, rest = weight.partition("://")
                repo = rest.split("@", 1)[0]
                if scheme in ("hf", "github") and provider.name not in users.setdefault(f"{scheme}/{repo}", []):
                    users[f"{scheme}/{repo}"].append(provider.name)
        groups: dict[tuple[str, str], int] = {}
        for rel, size in files.items():
            if rel in claimed or rel == "installed.json":
                continue
            parts = rel.split("/")
            if rel.endswith(".part"):
                key = ("partial", "Unfinished downloads")
            elif parts[0] in ("hf", "github") and len(parts) >= 4:
                key = ("builtin", f"{parts[1]}/{parts[2]}")
            else:
                key = ("other", "Other files")
            groups[key] = groups.get(key, 0) + size
        for (kind, name), size in groups.items():
            prefix = "hf" if (root / "hf" / name).exists() else "github"
            detail = ", ".join(users.get(f"{prefix}/{name}", [])) if kind == "builtin" else None
            items.append({"id": None, "kind": kind, "name": name, "detail": detail or None, "size": size})
        return sorted(items, key=lambda item: item["size"], reverse=True)

    def set_models_dir(self, path: Optional[str], move: bool) -> dict[str, Any]:
        """Point the store at ``path`` (None: the default), moving the current models there when asked."""
        if models_folder.mover.running:
            raise ValueError("Models are already moving to a new folder.")
        if any(d.status in ("queued", "downloading") for d in self.downloads.downloads.values()):
            raise ValueError("Wait for the downloads to finish, or cancel them, before changing the models folder.")
        target = Path(path).expanduser() if path else models_folder.default_root()
        if os.path.normcase(os.path.abspath(target)) != os.path.normcase(os.path.abspath(self.store.root)):
            models_folder.validate(target)
            if move and self.store.root.exists() and any(self.store.root.iterdir()):
                models_folder.mover.start(target)
            else:
                models_folder.set_models_root(target)
        return self.storage()


    # -- runtimes ----------------------------------------------------------------------------------

    def runtimes(self) -> dict[str, Any]:
        native = []
        for package in PACKAGES.values():
            installed = self.packages.installed(package)
            native.append(
                {
                    "id": package.id,
                    "name": package.name,
                    "repository": package.repository,
                    "installed": installed.tag if installed else None,
                    "variant": installed.variant if installed else None,
                    "checked_at": installed.checked_at if installed else None,
                    "pinned": self.packages.pin(package),
                    "latest": self.latest_tag(package),
                    "kept": [{"tag": tag, "variant": variant} for tag, variant, _ in self.packages.local_installs(package)],
                    "on_demand": package.on_demand,
                }
            )
        import importlib.metadata as metadata

        python = []
        for dist, name in PYTHON_RUNTIMES:
            try:
                python.append({"id": dist, "name": name, "installed": metadata.version(dist)})
            except metadata.PackageNotFoundError:
                continue
        return {"native": native, "python": python, "offline": self.packages.offline}

    def releases(self, package_id: str) -> list[dict[str, Any]]:
        return self.packages.releases(_package(package_id), self._variants())

    def set_pin(self, package_id: str, tag: Optional[str]) -> dict[str, Any]:
        """Pins ``package_id`` to ``tag`` and installs it; ``None`` unpins and returns to the newest release."""
        package = _package(package_id)
        devices = discover().candidates()
        self.packages.set_pin(package, tag)
        if tag:
            self.packages.install_tag(package, tag, devices)
        else:
            self.packages.ensure(package, devices, force=True)
        return self.runtimes()

    def update(self, package_id: str) -> dict[str, Any]:
        package = _package(package_id)
        if self.packages.pin(package):
            raise ValueError(f"{package.name} is pinned to {self.packages.pin(package)}; unpin it to update")
        self._latest.pop(package.id, None)
        self.packages.ensure(package, discover().candidates(), force=True)
        return self.runtimes()


def _has_safetensors(facts: ModelFacts) -> bool:
    return any(f.endswith(".safetensors") for f in facts.files)


def _package(package_id: str):
    from urakatsu.runtimes.packages import PACKAGES

    try:
        return PACKAGES[package_id]
    except KeyError as error:
        raise ValueError(f"unknown runtime {package_id!r}") from error


def _trim_readme(text: str, limit: int = 12000) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    return text.strip()[:limit]
