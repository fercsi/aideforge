from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from aideforge.tools import ToolPolicy
from .llm import LlmConfig
from .mcp import McpServerConfig
from .tool import ToolConfig

__all__ = ["AgentConfig", "Content", "ContextConfig", "ContextInfo"]

DEFAULT_SYSTEM_PROMPT = (
    "You are my AI assistant that has no access to any tools. "
    "Please, try to answer all my questions as well as possible without using any."
)


class Content(BaseModel):
    """A container for resource content.

    Container may contain the content itself or a path to the resource. This
    object contains also information on what type of resource is associated
    with it.
    """

    content: str | list[str | dict[Any, Any]] | None = Field(
        default=None,
        description=(
            "Direct text content. If provided, takes precedence over *path* "
            "and *handle*."
        ),
    )
    path: str | None = Field(
        default=None,
        description=(
            "A content path to read text content from. This path can be "
            "a relative path to the storage path in case of fs (or even "
            "absolute one), a path stored in a database, etc."
        ),
    )
    type: str = Field(
        default="Instruction",
        description=(
            'The type label of the content (e.g. ``"Instruction"``), used '
            "for tagging when embedded in a system prompt."
        ),
    )

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> Self:
        """Check if exactly one field is set, and the other is None"""
        if (self.content is not None) == (self.path is not None):
            raise ValueError("Exactly one of content or path must be defined")
        return self


class ContextConfig(BaseModel):
    """Configuration for context window management of an LLM."""

    max_input_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of input tokens. If ``None``, filled "
            "automatically from the model's profile during instantiation."
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of output tokens. If ``None``, filled "
            "automatically from the model's profile during instantiation."
        ),
    )


class ContextInfo(ContextConfig):
    """Extended context information including token usage statistics."""

    current_tokens: int | None = Field(
        default=None, description=("Current context content size in tokens")
    )
    last_input_tokens: int | None = Field(
        default=None, description=("Number of last prompt's input tokens.")
    )
    last_output_tokens: int | None = Field(
        default=None, description=("Number of last prompt's output tokens.")
    )
    last_total_tokens: int | None = Field(
        default=None, description=("Total number of last prompt's tokens.")
    )
    last_reasoning_tokens: int | None = Field(
        default=None, description=("Number of last prompt's reasoning tokens.")
    )


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""

    kind: Literal["agent"] = Field(
        default="agent",
        description='The type identifier, always "agent".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    name: str = Field(
        default="default",
        description="Unique name identifying the agent.",
    )
    description: str = Field(
        default="General agent for common question & answer chat",
        description="Human-readable description of the agent's purpose.",
    )
    default_tool_policy: ToolPolicy = Field(
        default=ToolPolicy.ASK,
        description="Default policy for tool access (ASK, ALLOW, DENY, INHERIT).",
    )
    tools: dict[str, ToolConfig] = Field(
        default_factory=dict,
        description="Dictionary of tool configurations keyed by tool name.",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt that defines the agent's behavior.",
    )
    resources: list[str | Content] = Field(
        default_factory=list,
        description="List of resources (paths or Content objects) available to the agent.",
    )
    welcome_message: str = Field(
        default="",
        description="Message shown when the agent is activated.",
    )
    llm: LlmConfig | None = Field(
        default=None,
        description="LLM configuration for this agent.",
    )
    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list,
        description="List of MCP server configurations.",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context window configuration.",
    )

    # Metadata
    # TODO: Agent scope: both
    scope: ClassVar[Literal["global"]] = "global"
    """*Model metadata*.
    Storage scope. ``"global"`` means agents are shared across users.
    """
    storage_name: ClassVar[str] = "{struct.name}"
    """*Model metadata*.
    Format string for storage name (agent's name), interpolated with the
    struct instance.
    """
    kinds: ClassVar[Literal["agents"]] = "agents"
    """*Model metadata*.
    Name for grouping agents (e.g. sub-directory).
    """
