import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pydantic import ValidationError

from aideforge.agent.blueprint import AgentBlueprint
from aideforge.structs.agent import AgentConfig, Content
from aideforge.structs.llm import LlmConfig
from aideforge.structs.session import AgentState
from aideforge.tools import ToolPolicy, PreconfiguredTool
from aideforge.abc.storage import AbstractStorage
from aideforge.utils import create_id, IdType
from langchain.messages import SystemMessage
from langchain.tools import BaseTool


class MockBaseTool(BaseTool):
    name: str = "mock_tool"
    description: str = "Mock tool"
    
    def _run(self, *args, **kwargs):
        return "mock result"


def _create_mock_tool_class(name="mock_tool"):
    """Create a mock tool class for testing."""
    code = f'''
class DynamicTool(BaseTool):
    name: str = "{name}"
    description: str = "Mock {name} tool"
    
    def _run(self, *args, **kwargs):
        return "mock result"
'''
    namespace = {"BaseTool": BaseTool}
    exec(code, namespace)
    return namespace["DynamicTool"]


class TestAgentBlueprint:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.agent_config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Test system prompt"
        )

    def test_creation_with_config_dict(self):
        bp = AgentBlueprint(config={"name": "test_agent", "llm": self.llm})
        assert bp._config.name == "test_agent"
        assert bp._config.llm == self.llm

    def test_creation_with_agent_config(self):
        bp = AgentBlueprint(config=self.agent_config)
        assert bp._config.name == "test_agent"

    def test_creation_with_none_config(self):
        bp = AgentBlueprint(config=None)
        assert bp._config.name == "default"
        assert bp._config.llm is None

    def test_creation_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid agent configuration"):
            AgentBlueprint(config="invalid")

    def test_tool_pool_registration(self):
        mock_tool_class = _create_mock_tool_class("mock_tool")
        
        bp = AgentBlueprint(
            config=self.agent_config,
            tool_pool=[mock_tool_class]
        )
        assert "mock_tool" in bp._tools

    def test_prepare_llm_config_with_agent_llm(self):
        bp = AgentBlueprint(config=self.agent_config)
        result = bp._prepare_llm_config(None, force_llm=False)
        assert result == self.llm

    def test_prepare_llm_config_with_force_llm(self):
        bp = AgentBlueprint(config=self.agent_config)
        external_llm = LlmConfig(model="gpt-3.5", provider="openai")
        result = bp._prepare_llm_config(external_llm, force_llm=True)
        assert result == external_llm

    def test_prepare_llm_config_no_agent_llm_no_external_raises(self):
        bp = AgentBlueprint(config=AgentConfig(name="test"))
        with pytest.raises(TypeError, match="missing required argument 'model'"):
            bp._prepare_llm_config(None, force_llm=False)

    def test_prepare_llm_config_no_agent_llm_with_external(self):
        bp = AgentBlueprint(config=AgentConfig(name="test"))
        external_llm = LlmConfig(model="gpt-4", provider="openai")
        result = bp._prepare_llm_config(external_llm, force_llm=False)
        assert result == external_llm

    @patch('aideforge.agent.blueprint.AgentRuntime')
    def test_instantiate(self, mock_runtime):
        bp = AgentBlueprint(config=self.agent_config)
        mock_runtime_instance = MagicMock()
        mock_runtime.return_value = mock_runtime_instance
        
        result = bp.instantiate()
        
        assert result == mock_runtime_instance
        mock_runtime.assert_called_once()

    @patch('aideforge.agent.blueprint.AgentRuntime')
    def test_instantiate_with_force_llm(self, mock_runtime):
        bp = AgentBlueprint(config=self.agent_config)
        external_llm = LlmConfig(model="gpt-3.5", provider="openai")
        mock_runtime_instance = MagicMock()
        mock_runtime.return_value = mock_runtime_instance
        
        result = bp.instantiate(llm=external_llm, force_llm=True)
        
        assert result == mock_runtime_instance

    @patch('aideforge.agent.blueprint.AgentRuntime')
    def test_restore(self, mock_runtime):
        bp = AgentBlueprint(config=self.agent_config)
        agent_id = create_id("agent")
        state = AgentState(
            config=self.agent_config,
            threads=[create_id("thread")],
            llm=self.llm
        )
        mock_runtime_instance = MagicMock()
        mock_runtime.return_value = mock_runtime_instance
        
        result = bp.restore(agent_id=agent_id, state=state)
        
        assert result == mock_runtime_instance
        mock_runtime.assert_called_once()


class TestAgentBlueprintTools:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.agent_config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Test system prompt"
        )

    def test_prepare_tools_empty(self):
        bp = AgentBlueprint(config=self.agent_config)
        tools = bp._prepare_tools()
        assert tools is None

    def test_prepare_tools_with_tool_class(self):
        mock_tool_class = _create_mock_tool_class("test_tool")
        mock_instance = mock_tool_class()
        
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            tools={"my_tool": {"type": "test_tool", "policy": "allow"}}
        )
        bp = AgentBlueprint(config=config, tool_pool=[mock_tool_class])
        
        tools = bp._prepare_tools()
        assert len(tools) == 1
        # The tool is renamed to the user-given name "my_tool"
        assert tools[0].name == "my_tool"

    def test_prepare_tools_with_preconfigured_tool(self):
        mock_tool_class = _create_mock_tool_class("preconfigured_tool")
        mock_instance = mock_tool_class()
        preconfigured = PreconfiguredTool(tool=mock_tool_class)
        
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            tools={"my_tool": {"type": "preconfigured_tool", "policy": "allow"}}
        )
        bp = AgentBlueprint(config=config, tool_pool=[preconfigured])
        
        tools = bp._prepare_tools()
        assert len(tools) == 1

    def test_prepare_tools_with_custom_name(self):
        mock_tool_class = _create_mock_tool_class("original_name")
        mock_instance = mock_tool_class()
        
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            tools={"custom_name": {"type": "original_name", "policy": "allow"}}
        )
        bp = AgentBlueprint(config=config, tool_pool=[mock_tool_class])
        
        tools = bp._prepare_tools()
        assert len(tools) == 1
        assert tools[0].name == "custom_name"

    def test_prepare_tools_with_config(self):
        mock_tool_class = _create_mock_tool_class("configured_tool")
        mock_instance = mock_tool_class()
        
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            tools={"my_tool": {"type": "configured_tool", "policy": "allow", "config": {"setting": "value"}}}
        )
        bp = AgentBlueprint(config=config, tool_pool=[mock_tool_class])
        
        tools = bp._prepare_tools()
        assert len(tools) == 1

    def test_prepare_tools_missing_tool_skipped(self):
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            tools={"my_tool": {"type": "nonexistent", "policy": "allow"}}
        )
        bp = AgentBlueprint(config=config, tool_pool=[])
        
        tools = bp._prepare_tools()
        assert tools is None


class TestAgentBlueprintSystemMessage:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")

    def test_create_system_message_basic(self):
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Basic prompt"
        )
        bp = AgentBlueprint(config=config)
        
        msg = bp._create_system_message()
        assert msg.content == "Basic prompt"

    def test_create_system_message_with_resource_content(self):
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Base prompt",
            resources=[Content(content="Resource content", type="Instruction")]
        )
        bp = AgentBlueprint(config=config)
        
        msg = bp._create_system_message()
        assert "Base prompt" in msg.content
        assert "Resource content" in msg.content
        assert "<Instruction>" in msg.content

    def test_create_system_message_with_resource_path_no_storage_raises(self):
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Base prompt",
            resources=[Content(path="/some/path", type="Instruction")]
        )
        bp = AgentBlueprint(config=config, storage=None)
        
        with pytest.raises(RuntimeError, match="No storage specified"):
            bp._create_system_message()

    def test_create_system_message_with_resource_path_and_storage(self):
        mock_storage = MagicMock(spec=AbstractStorage)
        mock_storage.retrieve_resource.return_value = "Retrieved content"
        
        config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Base prompt",
            resources=[Content(path="/some/path", type="Instruction")]
        )
        bp = AgentBlueprint(config=config, storage=mock_storage)
        
        msg = bp._create_system_message()
        assert "Base prompt" in msg.content
        assert "Retrieved content" in msg.content
        mock_storage.retrieve_resource.assert_called_once_with("/some/path")


class TestAgentBlueprintInterruptOn:
    def setup_method(self):
        self.llm = LlmConfig(model="gpt-4", provider="openai")

    def test_create_interrupt_on_none_tools(self):
        config = AgentConfig(name="test_agent", llm=self.llm)
        bp = AgentBlueprint(config=config)
        
        result = bp._create_interrupt_on(None)
        assert result is None

    def test_create_interrupt_on_with_tools(self):
        config = AgentConfig(name="test_agent", llm=self.llm)
        bp = AgentBlueprint(config=config)
        
        mock_tool1 = _create_mock_tool_class("tool1")()
        mock_tool2 = _create_mock_tool_class("tool2")()
        
        result = bp._create_interrupt_on([mock_tool1, mock_tool2])
        
        assert result is not None
        assert "tool1" in result
        assert "tool2" in result
        assert result["tool1"]["allowed_decisions"] == ["approve", "reject"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])