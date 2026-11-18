from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from aideforge.tools import ToolPolicy
from .abc import BaseStruct

__all__ = ["ToolModeConfig", "ToolConfig"]


class ToolModeConfig(BaseModel):
    r"""Configuration for a tool mode or sub-command.

    Examples
    --------
    >>> class MyShellConfig(BaseModel):
    ...   default_policy: ToolPolicy = ToolPolicy.DENY
    ...   show_warnings: bool = False
    ...   show_errors: bool = True
    ...   commands: dict[str, ToolModeConfig] = Field(default_factory=dict)
    ...
    >>> class MyShellTool(BaseTool):
    ...   def __init__(self, config: MyShellConfig | dict[str, Any]):
    ...   ...
    ...
    >>> agent_blueprint = AgentBlueprint(
    ...   config={
    ...     ...
    ...     "tools": {
    ...       "myshell": {
    ...         "policy": "allow",
    ...         "config": {
    ...           "show_errors": False,
    ...           "commands": {
    ...             "ls": {"policy": "allow"},
    ...             "rm": {"policy": "ask"},
    ...             ...
    ...           }
    ...         }
    ...       }
    ...     }
    ...     ...
    ...   }
    ... )
    """

    type: str | None = Field(
        default=None,
        description="Type identifier for the tool or tool mode.",
    )
    policy: ToolPolicy = Field(
        default=ToolPolicy.INHERIT,
        description=(
            "Access policy for this tool mode (INHERIT, ALLOW, ASK, DENY)."
        ),
    )
    config: Any = Field(
        default=None,
        description=(
            "Tool-specific configuration settings. Can be any structure the "
            "tool understands."
        ),
    )


class ToolConfig(BaseStruct, ToolModeConfig):
    """Persistent configuration for a tool."""

    kind: Literal["tool"] = Field(
        default="tool",
        description='The type identifier, always "tool".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    name: str = Field(..., description="Unique name identifying the tool.")

    # Metadata
    scope: ClassVar[Literal["global"]] = "global"
    """*Model metadata*.
    Storage scope. ``"global"`` means tools are shared across users.
    """
    storage_name: ClassVar[str] = "{struct.name}"
    """*Model metadata*.
    Format string for storage name, interpolated with the struct instance.
    """
    kinds: ClassVar[Literal["tools"]] = "tools"
    """*Model metadata*.
    Sub-directory for grouping tools.
    """
