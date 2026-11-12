"""Pydantic models used by the orchestrator (router)."""

from typing import List, Literal

from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    """A single agent task with targeted query."""

    source: Literal["search_agent_node", "planner_agent_node"]
    user_query: str
    focus: str = ""  # What aspect this agent should focus on


class ClassificationResult(BaseModel):
    """Router output - supports MULTIPLE parallel tasks."""

    tasks: List[AgentTask] = Field(
        description="List of agents to invoke with their targeted queries"
    )
    requires_synthesis: bool = Field(
        default=False,
        description="Whether multiple agents are being used and synthesis is needed",
    )
