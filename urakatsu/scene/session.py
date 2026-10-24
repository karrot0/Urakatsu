"""Scene session: commits patches, keeps undo/redo, persists to disk.

Storage splits as upstream does: the scene state is one JSON document written to
the inactive of two slots and then published atomically; blobs are immutable,
content-addressed files written before the state that references them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pydantic import TypeAdapter

from urakatsu.core import files

from . import components as c
from .scene import BlobReader, Entity, PageState, Patch, Relation, SceneError, Snapshot, StalePatch, State

UNDO_LIMIT = 64
BLOB_WRITERS = 8
_ORIGIN = TypeAdapter(c.Origin)


class BlobStore(BlobReader):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, bytes] = {}

    @staticmethod
    def address(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=20).hexdigest()

    def path(self, blob: str) -> Path:
        return self.root / blob[:2] / blob

    def write(self, data: bytes) -> str:
        blob = self.address(data)
        path = self.path(blob)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Commits write blobs outside the session lock, so two may land the same content at once.
            tmp = path.with_name(f"{blob}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            try:
                os.replace(tmp, path)
            except PermissionError:
                # Windows will not replace a file another writer has open; that writer wrote the same bytes.
                tmp.unlink(missing_ok=True)
                if not path.exists():
                    raise
        return blob

    def read(self, blob: str) -> bytes:
        cached = self._cache.get(blob)
        if cached is not None:
            return cached
        data = self.path(blob).read_bytes()
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[blob] = data
        return data


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.root / "blobs")
        self._lock = threading.RLock()
        self._state = self._open()
        self._undo: list[tuple[State, State]] = []
        self._redo: list[tuple[State, State]] = []

    # -- access ----------------------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._state, self.blobs)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # -- commits ---------------------------------------------------------------------------------

    def _write_blobs(self, patch: Patch) -> None:
        pending = list(patch.pending_blobs.items())
        if len(pending) > 1:
            # A chapter import writes dozens of files; on slow disks (exFAT, HDDs) each create and
            # rename waits on the filesystem, so they go concurrently.
            with ThreadPoolExecutor(min(BLOB_WRITERS, len(pending)), thread_name_prefix="blob-write") as pool:
                written = list(pool.map(self.blobs.write, [data for _, data in pending]))
        else:
            written = [self.blobs.write(data) for _, data in pending]
        if any(address != blob for (blob, _), address in zip(pending, written)):
            raise SceneError("blob address mismatch")

    def commit(self, patch: Patch) -> Snapshot:
        # Blobs are immutable and content-addressed, so they are written before taking the lock:
        # a large import (a downloaded chapter) does not stall readers while its images land.
        self._write_blobs(patch)
        with self._lock:
            current = self._state
            if patch.base_revision != current.revision:
                # Derived work rebases explicitly and fails if anything it observed has changed.
                self._validate_observed(patch, current)
                if patch.edit is None:
                    raise StalePatch("patch is stale and cannot be replayed")
                patch = patch.edit.replay_onto(Snapshot(current, self.blobs)).finish()
                self._write_blobs(patch)
            pages = dict(current.pages)
            for page_id, value in patch.pages.items():
                if value is None:
                    pages.pop(page_id, None)
                else:
                    pages[page_id] = value
            new = State(
                revision=current.revision + 1,
                page_order=patch.page_order if patch.page_order is not None else list(current.page_order),
                pages=pages,
                project=patch.project if patch.project is not None else dict(current.project),
            )
            self._persist(new)
            self._undo.append((current, new))
            del self._undo[:-UNDO_LIMIT]
            self._redo.clear()
            self._state = new
            return Snapshot(new, self.blobs)

    def _validate_observed(self, patch: Patch, current: State) -> None:
        """Rejects a stale patch: everything it observed must be unchanged (Koharu's rebase rule)."""
        for (owner, entity, kind), observed in patch.observed.items():
            if not current.contains(entity):
                if observed is not None and observed != {}:
                    raise StalePatch(f"entity {entity} changed since the patch was made")
                continue
            target = current.entity(entity)
            if owner == "assets":
                now: object = {role: asset.blob for role, asset in target.assets.items()}
            else:
                value = target.components.get(kind)
                now = value.model_dump(mode="json") if value is not None else None
            if now != observed:
                raise StalePatch(f"{kind} on {entity} changed since the patch was made")

    def undo(self) -> Optional[Snapshot]:
        with self._lock:
            if not self._undo:
                return None
            before, after = self._undo.pop()
            restored = State(
                revision=self._state.revision + 1,
                page_order=list(before.page_order),
                pages=dict(before.pages),
                project=dict(before.project),
            )
            self._persist(restored)
            self._redo.append((before, after))
            self._state = restored
            return Snapshot(restored, self.blobs)

    def redo(self) -> Optional[Snapshot]:
        with self._lock:
            if not self._redo:
                return None
            before, after = self._redo.pop()
            restored = State(
                revision=self._state.revision + 1,
                page_order=list(after.page_order),
                pages=dict(after.pages),
                project=dict(after.project),
            )
            self._persist(restored)
            self._undo.append((before, after))
            self._state = restored
            return Snapshot(restored, self.blobs)

    # -- persistence -----------------------------------------------------------------------------

    def _slots(self) -> tuple[Path, Path]:
        return self.root / "state.a.json", self.root / "state.b.json"

    def _open(self) -> State:
        best: Optional[State] = None
        existing = [slot for slot in self._slots() if slot.exists()]
        for slot in existing:
            try:
                state = decode_state(json.loads(slot.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if best is None or state.revision > best.revision:
                best = state
        if best is None and existing:
            # Like Session::recover, an unreadable project is an error, never an empty project that
            # the next commit would write over.
            raise SceneError(f"project state is unreadable: {', '.join(str(slot) for slot in existing)}")
        return best or State()

    def _persist(self, state: State) -> None:
        a, b = self._slots()
        target = a if state.revision % 2 == 0 else b
        files.write_text(target, json.dumps(encode_state(state)))


# ---- encoding --------------------------------------------------------------------------------------


def encode_state(state: State) -> dict:
    return {
        "revision": state.revision,
        "page_order": state.page_order,
        "project": {kind: value.model_dump(mode="json") for kind, value in state.project.items()},
        "pages": {
            page_id: {
                "entities": {
                    entity.id: {
                        "parent": entity.parent,
                        "children": entity.children,
                        "components": {kind: value.model_dump(mode="json") for kind, value in entity.components.items()},
                        "assets": {role: asset.model_dump(mode="json") for role, asset in entity.assets.items()},
                    }
                    for entity in page.entities.values()
                },
                "relations": {
                    relation.id: {
                        "kind": relation.kind,
                        "source": relation.source,
                        "target": relation.target,
                        "origin": _ORIGIN.dump_python(relation.origin, mode="json"),
                    }
                    for relation in page.relations.values()
                },
            }
            for page_id, page in state.pages.items()
        },
    }


def decode_state(data: dict) -> State:
    state = State(revision=int(data["revision"]), page_order=list(data["page_order"]))
    for kind, value in data.get("project", {}).items():
        cls = c.COMPONENTS.get(kind)
        if cls is not None:
            state.project[kind] = cls.model_validate(value)
    for page_id, page in data["pages"].items():
        page_state = PageState(id=page_id)
        for entity_id, entity in page["entities"].items():
            components = {}
            for kind, value in entity["components"].items():
                cls = c.COMPONENTS.get(kind)
                if cls is None:
                    raise ValueError(f"unknown component kind {kind}")  # fail closed
                components[kind] = cls.model_validate(value)
            page_state.entities[entity_id] = Entity(
                id=entity_id,
                parent=entity["parent"],
                children=list(entity["children"]),
                components=components,
                assets={role: c.Asset.model_validate(asset) for role, asset in entity.get("assets", {}).items()},
            )
        for relation_id, relation in page["relations"].items():
            page_state.relations[relation_id] = Relation(
                id=relation_id,
                kind=relation["kind"],
                source=relation["source"],
                target=relation["target"],
                origin=_ORIGIN.validate_python(relation["origin"]),
            )
        state.pages[page_id] = page_state
    return state
