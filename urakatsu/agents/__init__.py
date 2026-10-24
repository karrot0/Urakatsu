"""Agents: chat assistants that work in the app through tools.

- ``agent.py``: an agent is instructions plus a toolset; register new ones with
  ``@register_agent``.
- ``tools.py`` / ``app_tools.py``: the tool contract and Urakatsu's own tools.
- ``backends/``: wire formats (OpenAI Chat Completions, Anthropic Messages) and
  ``chatgpt.py`` (Responses API through a ChatGPT sign-in).
- ``models.py``: which models can run an agent, and the backend for each.
- ``session.py``: conversations and the tool-calling loop.
- ``commands.py``: slash commands, which run one tool without a model.
- ``service.py``: the process-wide ``AgentService`` the HTTP routes use.
"""

from .agent import Agent, agent_classes, get_agent, register_agent
from .service import AgentService
from .tools import Tool, ToolContext, Toolset, tool

__all__ = ["Agent", "AgentService", "Tool", "ToolContext", "Toolset", "agent_classes", "get_agent", "register_agent", "tool"]
