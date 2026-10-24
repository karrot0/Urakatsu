"""What a given runtime release can load, read from that release's own source.

Nothing here is a hardcoded model list:

- llama.cpp: the architecture names in ``src/llama-arch.cpp`` (``LLM_ARCH_NAMES``) at the
  release tag. A GGUF loads when its ``general.architecture`` is one of them.
- stable-diffusion.cpp: the model families in the README's "Supported models" list at the
  release tag (each links a ``docs/<family>.md`` page), and ``sd-server``'s recipes
  (``examples/server/recipes/*.json``): each model's components and defaults.

A model that the installed release cannot load but the newest release can is reported as
``needs_update``; users who pin an older runtime keep working and see a warning instead.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from urakatsu.runtimes.packages import LLAMA_CPP, STABLE_DIFFUSION_CPP, Package, PackageManager

if TYPE_CHECKING:  # pragma: no cover
    from .recipes import Recipe

RAW = "https://raw.githubusercontent.com"
_ARCH = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}')
_FAMILY = re.compile(r"\[([^\]]+)\]\(\./docs/([A-Za-z0-9_.\-]+)\.md(?:#[^)]*)?\)")


@dataclass(frozen=True)
class Family:
    """A stable-diffusion.cpp model family, e.g. ``Qwen Image 2.1`` documented in ``qwen_image_2.1.md``."""

    name: str
    doc: str
    aliases: tuple[str, ...]  # normalized alternatives from the name ("flux2klein", "flux2dev", ...)


@dataclass(frozen=True)
class RuntimeSupport:
    package: str
    tag: str
    architectures: frozenset[str] = frozenset()  # llama.cpp
    families: tuple[Family, ...] = ()  # stable-diffusion.cpp
    recipes: tuple["Recipe", ...] = ()  # stable-diffusion.cpp (sd-server recipes)

    def family(self, doc: str) -> Optional[Family]:
        return next((f for f in self.families if f.doc == doc), None)

    def recipe(self, recipe_id: str) -> Optional["Recipe"]:
        return next((r for r in self.recipes if r.id == recipe_id), None)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", text.lower()).replace(".", "")


def parse_families(readme: str) -> tuple[Family, ...]:
    section = readme.split("Supported models", 1)[-1].split("Supported backends", 1)[0]
    families: list[Family] = []
    for name, doc in _FAMILY.findall(section):
        if doc in ("edit", "photo_maker", "ip_adapter", "taesd", "esrgan", "distilled_sd", "adetailer"):
            continue  # feature pages, not model families
        parts = re.split(r"\s*[/,]\s*|\s+or\s+", name)
        aliases = tuple(dict.fromkeys(a for a in (normalize(p) for p in parts) if len(a) >= 3))
        if aliases and all(f.doc != doc or f.name != name for f in families):
            families.append(Family(name=name, doc=doc, aliases=aliases))
    return tuple(families)


def parse_architectures(source: str) -> frozenset[str]:
    return frozenset(_ARCH.findall(source))


class SupportIndex:
    """Caches each release's support data next to its install (or under ``runtimes/<pkg>/support``)."""

    def __init__(self, packages: Optional[PackageManager] = None, transport: Optional[httpx.BaseTransport] = None):
        self.packages = packages or PackageManager()
        self._transport = transport
        self._memory: dict[tuple[str, str], RuntimeSupport] = {}
        self._lock = threading.Lock()

    def _path(self, package: Package, tag: str) -> Path:
        return self.packages.root / package.id / "support" / f"{tag}.json"

    def for_tag(self, package: Package, tag: str) -> Optional[RuntimeSupport]:
        key = (package.id, tag)
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        path = self._path(package, tag)
        support = _load(path, package.id, tag)
        # Support files written before recipes were read lack them; fetch those once.
        stale = support is not None and package is STABLE_DIFFUSION_CPP and not support.recipes
        if (support is None or stale) and not self.packages.offline:
            try:
                fetched = self._fetch(package, tag)
            except httpx.HTTPError:
                fetched = None
            if fetched is not None:
                support = fetched
                self._save(path, support)
        if support is None:
            return None
        if support is not None:
            with self._lock:
                self._memory[key] = support
        return support

    def _save(self, path: Path, support: RuntimeSupport) -> None:
        from .recipes import recipe_to_json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "architectures": sorted(support.architectures),
                    "families": [{"name": f.name, "doc": f.doc, "aliases": list(f.aliases)} for f in support.families],
                    "recipes": [recipe_to_json(r) for r in support.recipes],
                }
            ),
            encoding="utf-8",
        )

    def doc(self, package: Package, tag: str, doc: str) -> str:
        """A family's ``docs/<doc>.md`` at ``tag`` (cached beside the support file); empty when unavailable."""
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", doc):
            return ""
        path = self.packages.root / package.id / "support" / tag / f"{doc}.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass
        if self.packages.offline:
            return ""
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0, transport=self._transport) as client:
                text = client.get(f"{RAW}/{package.repository}/{tag}/docs/{doc}.md").raise_for_status().text
        except httpx.HTTPError:
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def installed(self, package: Package) -> Optional[RuntimeSupport]:
        installed = self.packages.installed(package)
        return self.for_tag(package, installed.tag) if installed else None

    def _fetch(self, package: Package, tag: str) -> RuntimeSupport:
        with httpx.Client(follow_redirects=True, timeout=30.0, transport=self._transport) as client:
            if package is LLAMA_CPP:
                source = client.get(f"{RAW}/{package.repository}/{tag}/src/llama-arch.cpp").raise_for_status().text
                return RuntimeSupport(package.id, tag, architectures=parse_architectures(source))
            readme = client.get(f"{RAW}/{package.repository}/{tag}/README.md").raise_for_status().text
            return RuntimeSupport(package.id, tag, families=parse_families(readme), recipes=self._recipes(client, package, tag))

    def _recipes(self, client: httpx.Client, package: Package, tag: str) -> tuple["Recipe", ...]:
        """``sd-server``'s built-in recipes at ``tag``; none for releases that predate them."""
        from .recipes import parse_recipe

        headers = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        response = client.get(f"https://api.github.com/repos/{package.repository}/git/trees/{tag}", params={"recursive": "1"}, headers=headers)
        if response.status_code != 200:
            return ()
        names = [item["path"] for item in response.json().get("tree", []) if re.fullmatch(r"examples/server/recipes/[^/]+\.json", item.get("path", ""))]

        def fetch(name: str):
            try:
                return parse_recipe(client.get(f"{RAW}/{package.repository}/{tag}/{name}").raise_for_status().json())
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                return None

        with ThreadPoolExecutor(max_workers=12) as pool:
            return tuple(r for r in pool.map(fetch, names) if r is not None)


def _load(path: Path, package: str, tag: str) -> Optional[RuntimeSupport]:
    from .recipes import parse_recipe

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return RuntimeSupport(
        package,
        tag,
        architectures=frozenset(data.get("architectures") or []),
        families=tuple(Family(f["name"], f["doc"], tuple(f["aliases"])) for f in data.get("families") or []),
        recipes=tuple(parse_recipe(r) for r in data.get("recipes") or []),
    )


def match_family(families: tuple[Family, ...], candidates: list[str]) -> Optional[Family]:
    """The family whose longest alias appears in the repo name or its base models."""
    # Repos spell out what family names abbreviate: "stable-diffusion-xl" is "SDXL".
    texts = [normalize(c).replace("stablediffusion", "sd") for c in candidates if c]
    best: Optional[tuple[int, Family]] = None
    for family in families:
        for alias in family.aliases:
            if any(alias in text for text in texts) and (best is None or len(alias) > best[0]):
                best = (len(alias), family)
    return best[1] if best else None


__all__ = ["Family", "RuntimeSupport", "SupportIndex", "match_family", "normalize", "parse_architectures", "parse_families", "STABLE_DIFFUSION_CPP", "LLAMA_CPP"]
