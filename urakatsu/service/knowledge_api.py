"""JSON routes for a project's knowledge (``urakatsu.pipeline.knowledge``), under ``/api/v1``.

``/projects/{id}/knowledge`` is the glossary and notes; ``.../knowledge/pages/{page_id}`` is
what retrieval picks for a page, so the user can see what the translator will get. Every
route is a plain ``def``: changes write the project state to disk.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from urakatsu.core.registry import registry
from urakatsu.core.stage import Stage
from urakatsu.pipeline import knowledge
from urakatsu.providers.translation.language import Language
from urakatsu.scene import SceneError, SourceText

from .workspace import Workspace


class EntriesBody(BaseModel):
    entries: list[knowledge.EntryInput]


class EntryPatch(BaseModel):
    kind: Optional[Literal["character", "place", "term", "note"]] = None
    source: Optional[str] = None
    aliases: Optional[list[str]] = None
    translation: Optional[str] = None
    text: Optional[str] = None
    pinned: Optional[bool] = None


class EnabledBody(BaseModel):
    enabled: bool


def _guard(fn, *args):
    try:
        return fn(*args)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
    except SceneError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


def knowledge_router(workspace: Workspace) -> APIRouter:
    router = APIRouter()

    def language() -> str:
        return Language.coerce(workspace.config.translation.target_language).tag

    def changed(project_id: str, value: object) -> object:
        workspace.store.touch(project_id)
        return value

    @router.get("/projects/{project_id}/knowledge")
    def get_knowledge(project_id: str):
        return _guard(lambda: knowledge.read(workspace.session(project_id).snapshot()))

    @router.post("/projects/{project_id}/knowledge/entries")
    def add_entries(project_id: str, body: EntriesBody):
        return changed(project_id, _guard(lambda: knowledge.add(workspace.session(project_id), body.entries, language())))

    @router.patch("/projects/{project_id}/knowledge/entries/{entry_id}")
    def update_entry(project_id: str, entry_id: str, body: EntryPatch):
        fields = body.model_dump(exclude_unset=True, exclude_none=True)
        return changed(project_id, _guard(lambda: knowledge.update(workspace.session(project_id), entry_id, fields, language())))

    @router.delete("/projects/{project_id}/knowledge/entries/{entry_id}", status_code=204)
    def remove_entry(project_id: str, entry_id: str):
        changed(project_id, _guard(lambda: knowledge.remove(workspace.session(project_id), [entry_id])))
        return Response(status_code=204)

    @router.put("/projects/{project_id}/knowledge/enabled")
    def set_enabled(project_id: str, body: EnabledBody):
        return changed(project_id, _guard(lambda: knowledge.set_enabled(workspace.session(project_id), body.enabled)))

    @router.get("/projects/{project_id}/knowledge/pages/{page_id}")
    def retrieved(project_id: str, page_id: str):
        from urakatsu.pipeline.stages.translation import retrieve  # the stage's imports are heavy; loaded on first use

        def pick():
            scene = workspace.session(project_id).snapshot()
            scene.page(page_id)
            targets = []
            for layer in scene.text_layers(page_id):
                content = scene.layer_content(layer)
                source = scene.component(content, SourceText)
                if source is not None and source.text.value.strip():
                    targets.append((content, source.text.value))
            provider = registry.get(Stage.TRANSLATION, workspace.config.provider_for(Stage.TRANSLATION))
            return asdict(retrieve(scene, page_id, targets, language(), provider.context_budget))

        return _guard(pick)

    return router


__all__ = ["knowledge_router"]
