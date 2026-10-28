from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from .common import ToolPolicy
from .llm import LlmConfig
from .mcp import McpServerConfig
from .tool import ToolConfig

__all__ = ["AgentConfig", "Content", "ContextConfig", "ContextInfo"]

DEFAULT_SYSTEM_PROMPT = (
    "You are my AI assistant that has no access to any tools. "
    "Please, try to answer all my questions as well as possible without using any."
)


class Content(BaseModel):
    """A container for content that can be loaded from multiple sources.

    Parameters
    ----------
    content : str or list[str | dict[Any, Any]] or None, optional
        Direct text content. If provided, takes precedence over *path*
        and *handle*.
    path : str or None, optional
        A content path to read text content from. This path can be a relative
        path to the storage path in case of fs (or even absolute one), a path
        stored in a database, etc.
    type : str, optional
        The type label of the content (e.g. ``"Instruction"``), used for
        tagging when embedded in a system prompt.

    Raises
    ------
    ValueError
        If :meth:`get_text` is called and no content source is available.
    """

    content: str | list[str | dict[Any, Any]] | None = Field(
        default=None,
        description="Direct content, which can be string or a dict if any",
    )
    path: str | None = Field(
        default=None, description="Path to identify document"
    )
    type: str = Field(
        default="Instruction",
        description="Type of the content, like Instruction, Guidance, Soul, Memory...",
    )

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> Self:
        # Check that exactly one field is set, and the other is None
        if (self.content is not None) == (self.path is not None):
            raise ValueError("Exactly one of content or path must be defined")
        return self


class ContextConfig(BaseModel):
    """Configuration for context window management of an LLM.

    Attributes
    ----------
    max_input_tokens : int or None, optional
        Maximum number of input tokens. If ``None``, filled automatically
        from the model's profile during instantiation.
    max_output_tokens : int or None, optional
        Maximum number of output tokens. If ``None``, filled automatically
        from the model's profile during instantiation.
    """

    max_input_tokens: int | None = None  # filled automatically
    max_output_tokens: int | None = None  # filled automatically


class ContextInfo(ContextConfig):
    current_tokens: int | None = None
    last_input_tokens: int | None = None
    last_output_tokens: int | None = None
    last_total_tokens: int | None = None
    last_reasoning_tokens: int | None = None


class AgentConfig(BaseModel):
    kind: Literal["agent"] = "agent"
    version: Literal[1] = 1
    name: str = "default"
    descrition: str = "General agent for common question & answer chat"
    default_tool_policy: ToolPolicy = ToolPolicy.ASK
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    resources: list[str | Content] = Field(default_factory=list)
    welcome_message: str = ""
    llm: LlmConfig | None = None
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    context: ContextConfig = Field(default_factory=ContextConfig)

    # Metadata
    # TODO: Agent scope: both
    scope: ClassVar[Literal["global"]] = "global"
    storage_name: ClassVar[str] = "{struct.name}"
    kinds: ClassVar[Literal["agents"]] = "agents"
