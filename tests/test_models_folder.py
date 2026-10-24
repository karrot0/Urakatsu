"""Choosing the models folder and moving installed models into it."""

from __future__ import annotations

import time

import pytest

from urakatsu.catalog.installed import InstalledModels
from urakatsu.runtimes import models_folder
from urakatsu.runtimes.store import ModelStore


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("URAKATSU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(models_folder, "_root", None)
    monkeypatch.setattr(models_folder, "mover", models_folder.Mover())
    return tmp_path


def _wait(mover: models_folder.Mover) -> models_folder.Move:
    deadline = time.time() + 10
    while mover.running and time.time() < deadline:
        time.sleep(0.02)
    assert mover.move is not None
    return mover.move


def test_default_and_chosen_root_apply_to_every_store(home):
    store = ModelStore()
    assert store.root == home / "home" / "models"
    models_folder.set_models_root(home / "elsewhere")
    assert store.root == home / "elsewhere"  # read live, not captured at construction
    assert InstalledModels(store).path == home / "elsewhere" / "installed.json"
    models_folder._root = None  # a new process reads storage.json
    assert ModelStore().root == home / "elsewhere"
    models_folder.set_models_root(None)
    assert ModelStore().root == home / "home" / "models"


def test_move_switches_only_after_every_file_arrives(home):
    source = models_folder.models_root()
    (source / "hf" / "org" / "repo" / "abc").mkdir(parents=True)
    (source / "hf" / "org" / "repo" / "abc" / "model.gguf").write_bytes(b"x" * 1000)
    (source / "installed.json").write_text('{"models": []}', encoding="utf-8")
    target = home / "new-models"
    models_folder.validate(target)
    move = _wait(models_folder.mover) if models_folder.mover.start(target) else None
    assert move and move.status == "completed" and move.done == move.total == 1000 + len('{"models": []}')
    assert models_folder.models_root() == target
    assert (target / "hf" / "org" / "repo" / "abc" / "model.gguf").read_bytes() == b"x" * 1000
    assert not (source / "hf").exists()


def test_failed_move_keeps_the_old_folder(home, monkeypatch):
    source = models_folder.models_root()
    (source / "a").mkdir(parents=True)
    (source / "a" / "one.bin").write_bytes(b"1")
    (source / "a" / "two.bin").write_bytes(b"2")
    target = home / "broken"
    calls = {"n": 0}
    real_replace = models_folder.os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(models_folder.os, "replace", flaky)
    monkeypatch.setattr(models_folder.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    models_folder.mover.start(target)
    move = _wait(models_folder.mover)
    assert move.status == "failed"
    assert models_folder.models_root() == source
    assert (source / "a" / "one.bin").read_bytes() == b"1"  # renamed file came back
    assert (source / "a" / "two.bin").read_bytes() == b"2"


def test_validate_rejects_nesting_and_relative_paths(home):
    current = models_folder.models_root()
    current.mkdir(parents=True)
    with pytest.raises(ValueError):
        models_folder.validate(current / "inner")
    with pytest.raises(ValueError):
        models_folder.validate(current.parent)
    with pytest.raises(ValueError):
        models_folder.validate(models_folder.Path("relative/models"))
