from typing import Annotated, Literal

from pydantic import Field

from .abc import BaseStruct

__all__ = [
    "McpHttpServerConfig",
    "McpServerConfig",
    "McpStdioServerConfig",
]


class McpHttpServerConfig(BaseStruct):
    """Configuration for an HTTP-based MCP (Model Context Protocol) server."""

    kind: Literal["mcp_server"] = Field(
        default="mcp_server",
        description='The type identifier, always "mcp_server".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    mcp_type: Literal["http"] = Field(
        default="http",
        description='The MCP transport type, always "http" for this class.',
    )
    name: str = Field(
        ..., description="Unique name identifying the MCP server."
    )
    url: str = Field(..., description="HTTP URL of the MCP server endpoint.")


class McpStdioServerConfig(BaseStruct):
    """Configuration for a stdio-based MCP (Model Context Protocol) server."""

    kind: Literal["mcp_server"] = Field(
        default="mcp_server",
        description='The type identifier, always "mcp_server".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    mcp_type: Literal["stdio"] = Field(
        default="stdio",
        description='The MCP transport type, always "stdio" for this class.',
    )
    name: str = Field(
        ..., description="Unique name identifying the MCP server."
    )
    command: str = Field(
        ...,
        description="Command to execute the MCP server (without arguments).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="List of arguments to pass to the command.",
    )


McpServerConfig = Annotated[
    McpHttpServerConfig | McpStdioServerConfig, Field(discriminator="mcp_type")
]
"""Configuration for an HTTP-based or stdio based MCP (Model Context Protocol)
server.
"""
