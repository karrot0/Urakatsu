"""Atomic saves while another thread reads the same file (Windows refuses to replace an open file)."""

import threading

from urakatsu.pipeline.projects import ProjectStore


def test_project_saves_survive_concurrent_reads(isolated_home):
    store = ProjectStore()
    project = store.create("p")
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            store._read_meta(project)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for n in range(500):
            store._write_meta(project, {**store._read_meta(project), "n": n})
    finally:
        stop.set()
        reader.join()
    assert store._read_meta(project)["n"] == 499
    assert not [p for p in (store.root / project).iterdir() if p.suffix == ".tmp"]
