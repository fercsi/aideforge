from typing import Annotated

from pydantic import BaseModel, Field

__all__ = ["McpServerConfig"]


class McpHttpServerConfig(BaseModel):
    kind: literal["mcp_server"] = "mcp_server"
    version: literal[1] = 1
    mcp_type: Literal["http"] = "http"
    name: str = Field(..., description="Name of the MCP")
    url: str = Field(..., description="URL of the HTTP based MCP")


class McpStdioServerConfig(BaseModel):
    kind: literal["mcp_server"] = "mcp_server"
    version: literal[1] = 1
    mcp_type: Literal["stdio"] = "stdio"
    name: str = Field(..., description="Name of the MCP")
    command: str = Field(
        ...,
        description="Command name of STDIO based MCP without any arguments.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Argument list to add to command, if any.",
    )


McpServerConfig = Annotated[
    McpHttpServerConfig | McpStdioServerConfig, Field(discriminator="mcp_type")
]
