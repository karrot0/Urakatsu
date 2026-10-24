"""The agents' process-wide owner: sign-in, models, conversations and running turns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .agent import agent_classes, get_agent
from .chatgpt import ChatGptAuth
from .commands import COMMANDS, parse as parse_command
from .image_edit import ChatGptImageEditor
from .models import ModelDirectory
from .session import ConversationStore, Turns
from .tools import ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.service.workspace import Workspace


class AgentService:
    def __init__(self, workspace: "Workspace", chatgpt: Optional[ChatGptAuth] = None, store: Optional[ConversationStore] = None):
        self.workspace = workspace
        self.chatgpt = chatgpt or ChatGptAuth()
        self.models = ModelDirectory(workspace, self.chatgpt)
        self.store = store or ConversationStore()
        self.turns = Turns(self.store)

    def agents(self) -> list[dict[str, Any]]:
        return [{"id": cls.id, "name": cls.name, "description": cls.description} for cls in agent_classes()]

    def conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.store.get(conversation_id)
        return {**conversation.to_json(), "running": self.turns.running(conversation_id)}

    async def send(
        self,
        conversation_id: Optional[str],
        text: str,
        model: str,
        agent: str = "assistant",
        project_id: Optional[str] = None,
        page_id: Optional[str] = None,
        selection: Optional[list[str]] = None,
    ) -> str:
        """Starts a turn (creating the conversation when ``conversation_id`` is empty); returns its id.

        A message that starts with ``/`` is a slash command: it runs without the model, on the
        open page and the text blocks ``selection`` names."""
        text = text.strip()
        if not text:
            raise ValueError("the message is empty")
        # Image edits use the ChatGPT account whichever model chats; it says so when signed out.
        context = ToolContext(self.workspace, project_id=project_id, page_id=page_id, image_editor=ChatGptImageEditor(self.chatgpt))
        parsed = parse_command(text)
        command = None
        if parsed is not None:
            backend = None
            command = (parsed[0], parsed[0].build(context, parsed[1], list(selection or [])))
        else:
            backend = await self.models.backend(model)
        if conversation_id:
            conversation = self.store.get(conversation_id)
            conversation.project_id = conversation.project_id or project_id
        else:
            conversation = self.store.create(agent, model, project_id)
        self.turns.start(conversation, text, get_agent(conversation.agent), backend, context, command)
        return conversation.id

    def commands(self) -> list[dict[str, str]]:
        return [command.describe() for command in COMMANDS.values()]

    async def close(self) -> None:
        self.chatgpt.cancel_login()
        await self.turns.close()
