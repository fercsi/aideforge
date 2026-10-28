from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from aideforge.utils import IdType, create_id
from .agent import AgentConfig
from .common import ToolPolicy

from .abc import BaseStruct
# from .event import SessionEvent
from .llm import LlmConfig

__all__ = ["AgentState", "SessionState"]


class AgentState(BaseModel):
    config: AgentConfig
    threads: list[IdType] = Field(..., description="List of agent's threads")
    llm: LlmConfig = Field(..., description="Current LLM used by the agent")
    tool_policy: dict[str, ToolPolicy] | None = Field(
        default=None, description="Policy information for each tools"
    )
    env: dict[str, Any] | None = Field(
        default=None,
        description="Environment and agent dependent extra information.",
    )


class SessionState(BaseStruct):
    # Tools themselves are not stored in session
    kind: Literal["session"] = "session"
    version: Literal[1] = 1
    session_id: IdType = Field(
        default_factory=lambda: create_id("session"),
        description="Unique ID of the session",
    )
    agents: dict[IdType, AgentState] = Field(
        default_factory=dict,
        description="Agents' current state",
    )
    current_agent: IdType | None = Field(
        default = None,
        description="ID of current agent"
    )
    history: list[str] = Field(
        default_factory=list,
        description="List of recent user inputs (prompts and commands)",
    )
    input_count: int = Field(
        default=0,
        description=(
            "Number of inputs within this session. "
            "The same as len(history) until history is truncated."
        ),
    )
#>    events: list[SessionEvent] = Field(default_factory=list)
    env: dict[str, Any] | None = Field(
        default=None,
        description="Environment dependent extra information like CWD.",
    )

    storage_path: ClassVar[str] = "%/sessions/{struct.session_id}"
    scope: ClassVar[Literal["user"]] = "user"
