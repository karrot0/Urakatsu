"""packaging/set_version.py, which the release PR runs to bump the version everywhere."""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("set_version", ROOT / "packaging" / "set_version.py")
set_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(set_version)


@pytest.mark.parametrize(("name", "pattern"), set_version.FILES)
def test_every_pattern_finds_the_current_version(name: str, pattern: str):
    """A reformatted file would break the release PR; each pattern must match the version once."""
    import urakatsu

    matches = re.findall(pattern, (ROOT / name).read_bytes().decode("utf-8"))
    assert matches == [urakatsu.__version__]


def test_sets_the_version_and_keeps_line_endings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in {name for name, _ in set_version.FILES}:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(pyproject.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(set_version, "ROOT", tmp_path)

    changed = set_version.set_version("12.34.56")

    assert sorted(changed) == sorted({name for name, _ in set_version.FILES})
    for name, pattern in set_version.FILES:
        assert re.findall(pattern, (tmp_path / name).read_bytes().decode("utf-8")) == ["12.34.56"]
    assert b"\r\n" in pyproject.read_bytes() and b"\n" not in pyproject.read_bytes().replace(b"\r\n", b"")
    assert set_version.set_version("12.34.56") == []
    with pytest.raises(ValueError):
        set_version.set_version("v1.0")
