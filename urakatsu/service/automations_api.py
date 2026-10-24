"""JSON routes for automations (``service.automations``), under ``/api/v1``.

``/automations/*`` holds what is shared (the queue, model profiles); ``/projects/{id}/automations``
a project's automations and run history. The routes are ``async``: the automation manager lives
on the service's event loop, beside the batches it runs.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .automations import AutomationBody, ModelProfile
from .workspace import Workspace


def automations_router(workspace: Workspace) -> APIRouter:
    router = APIRouter()
    automations = workspace.automations

    def guard(fn, *args):
        try:
            return fn(*args)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    def project(project_id: str) -> dict:
        return {
            "automations": [a.model_dump(mode="json") for a in automations.list(project_id)],
            "runs": [r.model_dump(mode="json") for r in automations.runs(project_id)],
        }

    @router.get("/automations/status")
    async def status():
        return automations.status()

    @router.post("/automations/runs/{run_id}/stop")
    async def stop(run_id: str):
        guard(automations.stop, run_id)
        return automations.status()

    @router.get("/automations/profiles")
    async def profiles():
        return {"profiles": [p.model_dump(mode="json") for p in automations.profiles()]}

    @router.post("/automations/profiles")
    async def create_profile(body: ModelProfile):
        body.id = ""
        return guard(automations.save_profile, body).model_dump(mode="json")

    @router.put("/automations/profiles/{profile_id}")
    async def update_profile(profile_id: str, body: ModelProfile):
        body.id = profile_id
        return guard(automations.save_profile, body).model_dump(mode="json")

    @router.delete("/automations/profiles/{profile_id}", status_code=204)
    async def delete_profile(profile_id: str):
        automations.delete_profile(profile_id)

    @router.get("/projects/{project_id}/automations")
    async def list_automations(project_id: str):
        return guard(project, project_id)

    @router.post("/projects/{project_id}/automations")
    async def create(project_id: str, body: AutomationBody):
        guard(automations.create, project_id, body)
        return guard(project, project_id)

    @router.put("/projects/{project_id}/automations/{automation_id}")
    async def update(project_id: str, automation_id: str, body: AutomationBody):
        guard(automations.update, project_id, automation_id, body)
        return guard(project, project_id)

    @router.delete("/projects/{project_id}/automations/{automation_id}")
    async def delete(project_id: str, automation_id: str):
        guard(automations.delete, project_id, automation_id)
        return guard(project, project_id)

    @router.post("/projects/{project_id}/automations/{automation_id}/run")
    async def run(project_id: str, automation_id: str):
        guard(automations.run_now, project_id, automation_id)
        return guard(project, project_id)

    @router.post("/projects/{project_id}/automations/{automation_id}/reset")
    async def reset(project_id: str, automation_id: str):
        guard(automations.reset, project_id, automation_id)
        return guard(project, project_id)

    return router


__all__ = ["automations_router"]
