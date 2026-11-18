from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from aideforge.tools import ToolPolicy
from aideforge.utils import IdType, create_id
from .agent import AgentConfig, ContextInfo
from .abc import BaseStruct

# from .event import SessionEvent
from .llm import LlmConfig

__all__ = ["AgentState", "SessionState"]


class AgentState(BaseModel):
    """Runtime state for an agent within a session."""

    config: AgentConfig = Field(..., description="The agent's configuration.")
    threads: list[IdType] = Field(
        ...,
        description="List of conversation thread IDs associated with this agent.",
    )
    llm: LlmConfig = Field(
        ..., description="Current LLM configuration being used by the agent."
    )
    context_info: ContextInfo = Field(
        default_factory=ContextInfo,
        description="Information about the current context window usage.",
    )
    tool_policy: dict[str, ToolPolicy] | None = Field(
        default=None,
        description="Per-tool access policies, if overridden from defaults.",
    )
    env: dict[str, Any] | None = Field(
        default=None,
        description="Environment and agent-specific extra information.",
    )


class SessionState(BaseStruct):
    """Persistent state for a chat session."""

    # Tools themselves are not stored in session
    kind: Literal["session"] = Field(
        default="session",
        description='The type identifier, always "session".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    session_id: IdType = Field(
        default_factory=lambda: create_id("session"),
        description="Unique identifier for the session.",
    )
    agents: dict[IdType, AgentState] = Field(
        default_factory=dict,
        description="Mapping of agent IDs to their runtime states.",
    )
    current_agent: IdType | None = Field(
        default=None,
        description="ID of the currently active agent.",
    )
    history: list[str] = Field(
        default_factory=list,
        description=(
            "List of recent user inputs (prompts and commands).\n"
            "\n"
            ".. note::\n"
            "   The history can be truncated, so the beginning of the"
            "   prompts may be missing. Truncation does not affect"
            "   ``input_count``. It always reflects the number of user"
            "   input messages submitted during the session."
        ),
    )
    input_count: int = Field(
        default=0,
        description=(
            "Total number of inputs in this session. "
            "Same as len(history) until history is truncated."
        ),
    )
    # list[SessionEvent] = Field(default_factory=list)
    env: dict[str, Any] | None = Field(
        default=None,
        description="Environment-dependent extra information like CWD.",
    )

    scope: ClassVar[Literal["user"]] = "user"
    """*Model metadata*.
    Storage scope. ``"user"`` means sessions are isolated per user.
    """
    storage_name: ClassVar[str] = "{struct.session_id}"
    """*Model metadata*.
    Format string for storage name, interpolated with the struct instance.
    """
    kinds: ClassVar["sessions"] = "sessions"
    """*Model metadata*.
    Sub-directory for grouping sessions.
    """
