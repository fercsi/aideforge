from Pydantic import BaseModel

from .abc import BaseStruct
from .common import ToolPolicy


class ToolModeConfig(BaseModel):
    """
    Examples
    --------
    >>> class MyShellConfig(BaseModel):
    ...   default_policy: ToolPolicy = ToolPolicy.DENY
    ...   show_warnings: bool = False
    ...   show_errors: bool = True
    ...   commands: dict[str, ToolModeConfig] = Field(default_factory=dict)
    >>> class MyShellTool(BaseTool):
    ...   def __init__(self, config: MyShellConfig | dict[str, Any]):
    ...   ...
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

    name: str | None = Field(
        default=None,
        desvription="Name of the tool",
    )
    policy: ToolPolicy = Field(
        default=ToolPolicy.ALLOW,
        description="Access policy of the tool",
    )
    config: Any = Field(
        default=None,
        description="Tool's configuration settings. It can be anything"
        " the tool can process including a dict or pydantic model",
    )


class ToolConfig(BaseStruct, ToolModeConfig):
    kind: Literal["tool"] = "tool"
    version: Literal[1] = 1

    # Metadata
    scope: ClassVar[Literal["global"]] = "global"
    storage_name: ClassVar[str] = "{struct.name}"
    kinds: ClassVar[Literal["tools"]] = "tools"
