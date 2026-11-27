import pytest
from pydantic import ValidationError

from aideforge.structs.agent import Content, ContextConfig, ContextInfo, AgentConfig


class TestContent:
    def test_creation_with_content(self):
        c = Content(content="hello", path=None)
        assert c.content == "hello"
        assert c.path is None
        assert c.type == "Instruction"

    def test_creation_with_path(self):
        c = Content(content=None, path="/path/to/file")
        assert c.content is None
        assert c.path == "/path/to/file"
        assert c.type == "Instruction"

    def test_custom_type(self):
        c = Content(content="hello", type="CustomType")
        assert c.type == "CustomType"

    def test_both_content_and_path_raises(self):
        with pytest.raises(ValidationError, match="Exactly one of content or path must be defined"):
            Content(content="hello", path="/path")

    def test_neither_content_nor_path_raises(self):
        with pytest.raises(ValidationError, match="Exactly one of content or path must be defined"):
            Content()

    def test_list_content(self):
        c = Content(content=["item1", "item2", {"key": "value"}])
        assert c.content == ["item1", "item2", {"key": "value"}]


class TestContextConfig:
    def test_default_values(self):
        cc = ContextConfig()
        assert cc.max_input_tokens is None
        assert cc.max_output_tokens is None

    def test_with_values(self):
        cc = ContextConfig(max_input_tokens=4096, max_output_tokens=1024)
        assert cc.max_input_tokens == 4096
        assert cc.max_output_tokens == 1024


class TestContextInfo:
    def test_inherits_context_config(self):
        ci = ContextInfo(max_input_tokens=4096, max_output_tokens=1024)
        assert ci.max_input_tokens == 4096
        assert ci.max_output_tokens == 1024

    def test_token_fields_default_none(self):
        ci = ContextInfo()
        assert ci.current_tokens is None
        assert ci.last_input_tokens is None
        assert ci.last_output_tokens is None
        assert ci.last_total_tokens is None
        assert ci.last_reasoning_tokens is None

    def test_with_token_values(self):
        ci = ContextInfo(
            current_tokens=100,
            last_input_tokens=50,
            last_output_tokens=30,
            last_total_tokens=80,
            last_reasoning_tokens=10
        )
        assert ci.current_tokens == 100
        assert ci.last_input_tokens == 50
        assert ci.last_output_tokens == 30
        assert ci.last_total_tokens == 80
        assert ci.last_reasoning_tokens == 10


class TestAgentConfig:
    def test_default_values(self):
        ac = AgentConfig()
        assert ac.kind == "agent"
        assert ac.version == 1
        assert ac.name == "default"
        assert ac.description == "General agent for common question & answer chat"
        from aideforge.tools import ToolPolicy
        assert ac.default_tool_policy == ToolPolicy.ASK
        assert ac.tools == {}
        assert ac.system_prompt == "You are my AI assistant that has no access to any tools. Please, try to answer all my questions as well as possible without using any."
        assert ac.resources == []
        assert ac.welcome_message == ""
        assert ac.llm is None
        assert ac.mcp_servers == []
        assert ac.context.max_input_tokens is None
        assert ac.context.max_output_tokens is None

    def test_custom_values(self):
        ac = AgentConfig(
            name="custom_agent",
            description="Custom agent",
            default_tool_policy="allow",
            system_prompt="Custom prompt",
            welcome_message="Welcome!",
        )
        assert ac.name == "custom_agent"
        assert ac.description == "Custom agent"
        assert ac.default_tool_policy == "allow"
        assert ac.system_prompt == "Custom prompt"
        assert ac.welcome_message == "Welcome!"

    def test_metadata(self):
        assert AgentConfig.scope == "global"
        assert AgentConfig.storage_name == "{struct.name}"
        assert AgentConfig.kinds == "agents"

    def test_with_tools(self):
        ac = AgentConfig(tools={"tool1": {"name": "tool1", "type": "test", "policy": "allow"}})
        assert "tool1" in ac.tools
        assert ac.tools["tool1"].policy == "allow"

    def test_with_mcp_servers(self):
        ac = AgentConfig(
            mcp_servers=[
                {"mcp_type": "http", "name": "server1", "url": "http://localhost:8000"},
                {"mcp_type": "stdio", "name": "server2", "command": "python", "args": ["server.py"]}
            ]
        )
        assert len(ac.mcp_servers) == 2
        assert ac.mcp_servers[0].mcp_type == "http"
        assert ac.mcp_servers[0].name == "server1"
        assert ac.mcp_servers[1].mcp_type == "stdio"
        assert ac.mcp_servers[1].name == "server2"