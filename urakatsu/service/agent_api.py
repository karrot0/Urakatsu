"""JSON routes for agents under ``/api/v1/agent``.

``/agent/models`` lists what can run an agent, ``/agent/auth/chatgpt*`` signs in with
ChatGPT, and ``/agent/conversations*`` holds the chats. Sending a message starts a turn
in the background; ``.../events`` streams its events as server-sent events, replaying
the turn so far, so the panel can re-attach after it was closed.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from urakatsu.agents import AgentService
from urakatsu.agents.backends.base import AgentError


class MessageBody(BaseModel):
    text: str
    model: str
    agent: str = "assistant"
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    page_id: Optional[str] = None
    # Text blocks selected in the editor (layer ids), for slash commands.
    selection: list[str] = []


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except KeyError as error:
        raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
    except (ValueError, AgentError) as error:
        raise HTTPException(400, str(error)) from error


def agent_router(agents: AgentService) -> APIRouter:
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.get("/agents")
    def list_agents():
        return agents.agents()

    @router.get("/commands")
    def list_commands():
        """The slash commands the message box offers."""
        return agents.commands()

    @router.get("/models")
    async def list_models(refresh: bool = False):
        """Answers from the saved listing at once; ``stale`` asks the caller to refresh in the background."""
        models, stale = await agents.models.list(refresh)
        return {"models": [m.model_dump() for m in models], "stale": stale}

    @router.get("/auth/chatgpt")
    def chatgpt_status():
        return agents.chatgpt.status()

    @router.post("/auth/chatgpt/login")
    def chatgpt_login():
        url = _guard(agents.chatgpt.start_login)
        return {"url": url}

    @router.post("/auth/chatgpt/cancel", status_code=204)
    def chatgpt_cancel():
        agents.chatgpt.cancel_login()

    @router.post("/auth/chatgpt/logout", status_code=204)
    def chatgpt_logout():
        agents.chatgpt.logout()

    @router.get("/conversations")
    def list_conversations(project_id: Optional[str] = Query(None)):
        return agents.store.list(project_id)

    @router.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        return _guard(agents.conversation, conversation_id)

    @router.delete("/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str):
        agents.turns.stop(conversation_id)
        _guard(agents.store.delete, conversation_id)

    @router.post("/messages")
    async def send_message(body: MessageBody):
        try:
            conversation_id = await agents.send(
                body.conversation_id, body.text, body.model, body.agent, body.project_id, body.page_id, body.selection
            )
        except KeyError as error:
            raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
        except (ValueError, AgentError) as error:
            raise HTTPException(400, str(error)) from error
        return {"conversation_id": conversation_id}

    @router.post("/conversations/{conversation_id}/stop", status_code=204)
    def stop(conversation_id: str):
        agents.turns.stop(conversation_id)

    @router.get("/conversations/{conversation_id}/events")
    async def events(conversation_id: str):
        async def stream():
            async for event in agents.turns.events(conversation_id):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return router
