"""Mayo's bundled font catalog: download location, face resolution, and the offline fallback."""

import json
import shutil
from pathlib import Path

import pytest

from urakatsu.renderer import fonts as fonts_module
from urakatsu.renderer.fonts import FONT_INDEX, FONT_REVISION, Fonts, catalog_file, download_catalog
from urakatsu.runtimes.store import HuggingFaceFile, ModelStore

SYSTEM_FONT = Path("C:/Windows/Fonts/arial.ttf")


def index_for(family: str, file_name: str, post_script_name: str) -> dict:
    return {
        "schema_version": 3,
        "families": [
            {
                "family_name": family,
                "weights": [
                    {
                        "weight": 400,
                        "fonts": [{"file_name": file_name, "postscript_name": post_script_name, "width": 1.0, "style": "normal", "preview": {"file": "p.webp"}}],
                    }
                ],
            }
        ],
    }


def test_dataset_files_have_their_own_url_and_cache_folder(tmp_path):
    store = ModelStore(tmp_path, offline=True)
    dataset = HuggingFaceFile("mayocream/fonts", "index.json", "abc", repo_type="dataset")
    model = HuggingFaceFile("org/model", "w.onnx", "abc")
    assert dataset.url() == "https://huggingface.co/datasets/mayocream/fonts/resolve/abc/index.json"
    assert dataset.local_path(store) == tmp_path / "hf" / "datasets" / "mayocream" / "fonts" / "abc" / "index.json"
    # Model files keep the folder they always had.
    assert model.local_path(store) == tmp_path / "hf" / "org" / "model" / "abc" / "w.onnx"
    assert catalog_file(FONT_INDEX).revision == FONT_REVISION


@pytest.mark.skipif(not SYSTEM_FONT.exists(), reason="needs a Windows system font to stand in for a bundled face")
def test_a_bundled_family_resolves_from_the_downloaded_catalog(tmp_path, monkeypatch):
    # A cached snapshot: the store finds both files without the network.
    store = ModelStore(tmp_path, offline=True)
    index = catalog_file(FONT_INDEX).local_path(store)
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(index_for("Bundled Only Sans", "Bundled Only Sans.ttf", "BundledOnly-Regular")), encoding="utf-8")
    shutil.copy(SYSTEM_FONT, catalog_file("Bundled Only Sans.ttf").local_path(store))

    catalog = download_catalog(store)
    resources = Fonts(catalog)
    assert any(family.name == "Bundled Only Sans" for family in resources.families())
    resolved = resources.resolve(None, None, None, ["Bundled Only Sans"], "HELLO")
    assert resolved[0].post_script_name == "BundledOnly-Regular"


@pytest.mark.skipif(not SYSTEM_FONT.exists(), reason="needs Windows system fonts")
def test_a_face_that_cannot_download_falls_back(tmp_path):
    store = ModelStore(tmp_path, offline=True)
    index = catalog_file(FONT_INDEX).local_path(store)
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(index_for("Bundled Only Sans", "missing.ttf", "Missing-Regular")), encoding="utf-8")
    resolved = Fonts(download_catalog(store)).resolve(None, None, None, ["Bundled Only Sans"], "HELLO")
    assert resolved and resolved[0].post_script_name != "Missing-Regular"


def test_an_unreachable_catalog_means_system_fonts_and_a_later_retry(monkeypatch):
    calls = []

    def unavailable(store=None):
        calls.append(1)
        raise OSError("offline")

    monkeypatch.delenv("URAKATSU_FONT_CATALOG", raising=False)
    monkeypatch.setattr(fonts_module, "download_catalog", unavailable)
    resources = Fonts()
    assert resources.catalog is None and resources.catalog is None
    assert len(calls) == 1  # not retried on every lookup
    resources._catalog_retry_at = 0.0
    assert resources.catalog is None and len(calls) == 2
