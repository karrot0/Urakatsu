"""Request scope."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

from urakatsu.core.stage import Stage
from urakatsu.scene import Geometry, Relations, Snapshot

SEMANTIC_RELATIONS = {
    Relations.PRESENTS,
    Relations.RECOGNIZED_FROM,
    Relations.FITS_TO,
    Relations.FLOWS_IN,
    Relations.INSIDE,
}


@dataclass(frozen=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float

    def validate(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(v) for v in values) or self.width <= 0 or self.height <= 0:
            raise ValueError("region bounds must be finite and non-empty")

    def intersects(self, geometry: Geometry) -> bool:
        extents = geometry_extents(geometry)
        if extents is None:
            return False
        min_x, min_y, max_x, max_y = extents
        return min_x < self.x + self.width and max_x > self.x and min_y < self.y + self.height and max_y > self.y


@dataclass(frozen=True)
class ProjectScope:
    pass


@dataclass(frozen=True)
class PagesScope:
    pages: tuple[str, ...]


@dataclass(frozen=True)
class RegionScope:
    page: str
    bounds: Bounds


@dataclass(frozen=True)
class EntitiesScope:
    entities: tuple[str, ...]


Scope = Union[ProjectScope, PagesScope, RegionScope, EntitiesScope]


class NormalizedScope:
    def __init__(self, snapshot: Snapshot, scope: Scope, stages: list[Stage]):
        if isinstance(scope, EntitiesScope) and any(s in (Stage.DETECTION, Stage.INPAINTING) for s in stages):
            raise ValueError("detection and inpainting do not support entity-only scope")
        self.entities: Optional[set[str]] = None
        self._region: Optional[tuple[str, Bounds]] = None
        if isinstance(scope, ProjectScope):
            self.pages = snapshot.pages()
        elif isinstance(scope, PagesScope):
            if not scope.pages:
                raise ValueError("page scope is empty")
            requested = set(scope.pages)
            for page in requested:
                snapshot.page(page)
            self.pages = [page for page in snapshot.pages() if page in requested]
        elif isinstance(scope, RegionScope):
            snapshot.page(scope.page)
            scope.bounds.validate()
            self.pages = [scope.page]
            self._region = (scope.page, scope.bounds)
        else:
            if not scope.entities:
                raise ValueError("entity scope is empty")
            page_set = {containing_page(snapshot, entity) for entity in scope.entities}
            hierarchy: set[str] = set()
            for entity in scope.entities:
                hierarchy.update(snapshot.subtree(entity))
            self.entities = semantic_closure(snapshot, hierarchy)
            self.pages = [page for page in snapshot.pages() if page in page_set]

    def region(self, page: str) -> Optional[Bounds]:
        if self._region and self._region[0] == page:
            return self._region[1]
        return None


def semantic_closure(snapshot: Snapshot, entities: set[str]) -> set[str]:
    pending = deque(entities)
    while pending:
        entity = pending.popleft()
        for relation in snapshot.relations_from(entity) + snapshot.relations_to(entity):
            if relation.kind not in SEMANTIC_RELATIONS:
                continue
            for related in (relation.source, relation.target):
                if related not in entities:
                    entities.add(related)
                    pending.append(related)
    return entities


def contains_entity(
    snapshot: Snapshot,
    page: str,
    entities: Optional[set[str]],
    region: Optional[Bounds],
    entity: str,
) -> bool:
    if containing_page(snapshot, entity) != page:
        return False
    if entities is not None:
        return entity in entities
    if region is None:
        return True
    geometry = snapshot.component(entity, Geometry)
    return geometry is not None and region.intersects(geometry)


def containing_page(snapshot: Snapshot, entity: str) -> str:
    return snapshot.page_of(entity)


def geometry_extents(geometry: Geometry) -> Optional[tuple[float, float, float, float]]:
    if not geometry.points:
        return None
    xs = [p.x for p in geometry.points]
    ys = [p.y for p in geometry.points]
    return min(xs), min(ys), max(xs), max(ys)
