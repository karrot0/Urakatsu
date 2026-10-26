"""Sets Urakatsu's version in every file that records it.

    python packaging/set_version.py 0.2.0

The release PR (.github/release-please/index.cjs) runs this for the version it prepares, as
Koharu's runs ``cargo set-version``. Each file keeps its own line endings; nothing is resolved
again, so no network or toolchain is needed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, pattern whose group 1 is the version). Each pattern must match exactly once.
FILES = [
    ("pyproject.toml", r'(?m)^\[project\][^\[]*?^version = "([^"]+)"'),
    ("urakatsu/__init__.py", r'(?m)^__version__ = "([^"]+)"'),
    ("uv.lock", r'(?m)^name = "urakatsu"\r?\nversion = "([^"]+)"'),
    ("web/package.json", r'^\{\s*"name": "@urakatsu/web",\s*"version": "([^"]+)"'),
    ("web/package-lock.json", r'^\{\s*"name": "@urakatsu/web",\s*"version": "([^"]+)"'),
    ("web/package-lock.json", r'"packages": \{\s*"": \{\s*"name": "@urakatsu/web",\s*"version": "([^"]+)"'),
    ("packaging/launcher/Cargo.toml", r'(?m)^\[package\]\r?\nname = "urakatsu-launcher"\r?\nversion = "([^"]+)"'),
    ("packaging/launcher/Cargo.lock", r'(?m)^name = "urakatsu-launcher"\r?\nversion = "([^"]+)"'),
]


def set_version(version: str) -> list[str]:
    """Writes ``version`` into every file; returns the files that changed."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"not a version: {version!r} (expected X.Y.Z)")
    texts: dict[str, str] = {}
    for name, pattern in FILES:
        text = texts.get(name)
        if text is None:
            text = (ROOT / name).read_bytes().decode("utf-8")
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected one version, found {len(matches)}")
        start, end = matches[0].span(1)
        texts[name] = text[:start] + version + text[end:]
    changed = []
    for name, text in texts.items():
        path = ROOT / name
        if path.read_bytes() != text.encode("utf-8"):
            path.write_bytes(text.encode("utf-8"))
            changed.append(name)
    return changed


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python packaging/set_version.py X.Y.Z")
    for name in set_version(sys.argv[1]):
        print(name)


if __name__ == "__main__":
    main()
