import pytest
from pydantic import ValidationError

from aideforge.structs.mcp import McpHttpServerConfig, McpStdioServerConfig, McpServerConfig


class TestMcpHttpServerConfig:
    def test_creation(self):
        config = McpHttpServerConfig(name="server1", url="http://localhost:8000")
        assert config.kind == "mcp_server"
        assert config.version == 1
        assert config.mcp_type == "http"
        assert config.name == "server1"
        assert config.url == "http://localhost:8000"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            McpHttpServerConfig(url="http://localhost:8000")

    def test_missing_url_raises(self):
        with pytest.raises(ValidationError):
            McpHttpServerConfig(name="server1")


class TestMcpStdioServerConfig:
    def test_creation(self):
        config = McpStdioServerConfig(
            name="server2",
            command="python",
            args=["server.py", "--port", "8000"]
        )
        assert config.kind == "mcp_server"
        assert config.version == 1
        assert config.mcp_type == "stdio"
        assert config.name == "server2"
        assert config.command == "python"
        assert config.args == ["server.py", "--port", "8000"]

    def test_default_empty_args(self):
        config = McpStdioServerConfig(name="server2", command="python")
        assert config.args == []

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            McpStdioServerConfig(command="python")

    def test_missing_command_raises(self):
        with pytest.raises(ValidationError):
            McpStdioServerConfig(name="server2")


class TestMcpServerConfigUnion:
    def test_http_config(self):
        config: McpServerConfig = McpHttpServerConfig(name="http_server", url="http://localhost:8000")
        assert config.mcp_type == "http"
        assert config.url == "http://localhost:8000"
        assert not hasattr(config, "command")

    def test_stdio_config(self):
        config: McpServerConfig = McpStdioServerConfig(name="stdio_server", command="python", args=["server.py"])
        assert config.mcp_type == "stdio"
        assert config.command == "python"
        assert config.args == ["server.py"]
        assert not hasattr(config, "url")

    def test_discriminator_validation(self):
        from pydantic import TypeAdapter
        adapter = TypeAdapter(McpServerConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "kind": "mcp_server",
                "version": 1,
                "mcp_type": "http",
                "name": "test"
            })