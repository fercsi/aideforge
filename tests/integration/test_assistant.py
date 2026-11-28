import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from pydantic import ValidationError

from aideforge.assistant import Assistant
from aideforge.structs.agent import AgentConfig
from aideforge.structs.llm import LlmConfig
from aideforge.structs.session import SessionState
from aideforge.structs.settings import AssistantSettings, UiSettings
from aideforge.storage import MemoryStorage
from aideforge.tools import ToolPolicy
from aideforge.utils import create_id
from aideforge.abc.storage import AbstractStorage


class TestAssistantIntegration:
    @pytest.fixture
    def llm(self):
        return LlmConfig(model="gpt-4", provider="openai")

    @pytest.fixture
    def agent_config(self, llm):
        return AgentConfig(
            name="test_agent",
            llm=llm,
            system_prompt="Test prompt",
            tools={"test_tool": {"type": "test", "policy": "allow"}}
        )

    @pytest.fixture
    def assistant_settings(self, llm):
        return AssistantSettings(llm=llm)

    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    def test_full_assistant_setup(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assert assistant.storage == storage
        assert "test_agent" in assistant.agent_blueprints
        assert assistant.current_agent_name == "test_agent"

    def test_assistant_prepare_cleanup(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assistant.prepare()
        assert storage._checkpoint_store is not None
        
        assistant.cleanup()
        # cleanup called

    def test_session_management(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        # Create new session
        session_id = create_id("session")
        storage.store_struct(SessionState(session_id=session_id))
        
        assistant.change_session(str(session_id))
        assert assistant.session.session_id == session_id

    def test_agent_switching(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assistant.change_agent("test_agent")
        assert assistant.current_agent_name == "test_agent"

    def test_available_agents(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        agents = assistant.available_agents
        assert "test_agent" in agents

    def test_render_template(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        result = assistant.render_template("Hello {agent}")
        assert "test_agent" in result or "default" in result

    def test_settings_property(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assert assistant.settings is not None
        # Assistant settings are always registered
        assert assistant.settings.assistant is not None
        # UI settings may not be registered by default
        # The settings group allows accessing domains via getitem
        assert hasattr(assistant.settings, 'assistant')

    def test_description_property(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config],
            description="Test description"
        )
        
        assert assistant.description == "Test description"

    def test_register_commands(self, storage, assistant_settings, agent_config):
        from aideforge.abc.command import BaseCommand
        from aideforge.results import CommandResult, Success
        
        class MockCommand(BaseCommand):
            name = "mock"
            aliases = []
            description = "Mock command"
            
            def execute(self, args: list[str]) -> CommandResult:
                return Success()
        
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assistant.register_commands([MockCommand])
        assert "mock" in assistant.commands


class TestAssistantPromptProcessing:
    @pytest.fixture
    def llm(self):
        return LlmConfig(model="gpt-4", provider="openai")

    @pytest.fixture
    def agent_config(self, llm):
        return AgentConfig(
            name="test_agent",
            llm=llm,
            system_prompt="Test prompt"
        )

    @pytest.fixture
    def assistant_settings(self, llm):
        return AssistantSettings(llm=llm)

    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    @pytest.mark.asyncio
    async def test_prompt_processor_basic(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        # Mock the agent's aprompt_processor
        mock_agent = MagicMock()
        
        # Create async generator that yields a response
        async def mock_generator():
            from aideforge.results import LlmResponse
            yield LlmResponse(content="Test response")
        
        mock_agent.aprompt_processor = MagicMock(return_value=mock_generator())
        
        with patch.object(assistant, '_get_current_agent', return_value=mock_agent):
            assistant._session = SessionState(session_id=create_id("session"))
            assistant.current_agent_name = "test_agent"
            
            processor = assistant.aprompt_processor("test prompt")
            response = await anext(processor)
            
            assert hasattr(response, 'content')
            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_prompt_processor_with_action_request(self, storage, assistant_settings, agent_config):
        from aideforge.agent.types import ActionRequest, ApprovalDecision
        
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        mock_agent = MagicMock()
        
        async def mock_generator():
            from aideforge.agent.types import ActionRequest
            from aideforge.results import LlmResponse
            
            # First yield an action request
            yield ActionRequest(name="test_tool", args={"arg": "val"}, description="Test")
            # Then yield response after approval
            yield LlmResponse(content="Tool executed")
        
        mock_agent.aprompt_processor = MagicMock(return_value=mock_generator())
        
        with patch.object(assistant, '_get_current_agent', return_value=mock_agent):
            assistant._session = SessionState(session_id=create_id("session"))
            assistant.current_agent_name = "test_agent"
            
            processor = assistant.aprompt_processor("test prompt")
            
            # First response should be ActionRequest
            response = await anext(processor)
            assert isinstance(response, ActionRequest)
            
            # Send approval
            response = await processor.asend(ApprovalDecision.APPROVE)
            assert hasattr(response, 'content')
            assert response.content == "Tool executed"


class TestAssistantCommandProcessing:
    @pytest.fixture
    def llm(self):
        return LlmConfig(model="gpt-4", provider="openai")

    @pytest.fixture
    def agent_config(self, llm):
        return AgentConfig(
            name="test_agent",
            llm=llm,
            system_prompt="Test prompt"
        )

    @pytest.fixture
    def assistant_settings(self, llm):
        return AssistantSettings(llm=llm)

    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    def test_process_command_unknown(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        result = assistant._process_command("/unknown arg1 arg2")
        assert hasattr(result, 'status')
        assert result.status == "error"

    def test_process_command_empty(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        result = assistant._process_command("/")
        assert hasattr(result, 'status')
        assert result.status == "error"

    def test_process_command_help(self, storage, assistant_settings, agent_config):
        from aideforge.abc.command import BaseCommand
        from aideforge.results import CommandResult, CommandOutput
        
        class TestCommand(BaseCommand):
            name = "testcmd"
            aliases = []
            description = "Test command"
            
            def execute(self, args: list[str]) -> CommandResult:
                return CommandOutput(content="Test output")
        
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        assistant.register_commands([TestCommand])
        
        result = assistant._process_command("/testcmd")
        # Should execute test command
        assert hasattr(result, 'content')
        assert result.content == "Test output"


class TestAssistantStorageIntegration:
    @pytest.fixture
    def llm(self):
        return LlmConfig(model="gpt-4", provider="openai")

    @pytest.fixture
    def agent_config(self, llm):
        return AgentConfig(
            name="test_agent",
            llm=llm,
            system_prompt="Test prompt"
        )

    @pytest.fixture
    def assistant_settings(self, llm):
        return AssistantSettings(llm=llm)

    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    def test_session_persistence(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        # Reset mock to get clean count
        storage._structs.clear()
        storage._resources.clear()
        
        # Store session
        assistant._session = SessionState(session_id=create_id("session"), input_count=5)
        assistant._stored_input_count = 0
        assistant._store_session()
        
        # Verify stored
        sessions = storage.list_structs(SessionState)
        assert len(sessions) == 1

    def test_agent_state_persistence(self, storage, assistant_settings, agent_config):
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent_config]
        )
        
        # Agent configs are registered in memory (blueprints), not stored by default
        # Verify they are available in blueprints
        assert "test_agent" in assistant.agent_blueprints
        
        # Cleanup stores session and settings
        assistant.cleanup()
        
        # Verify settings stored
        settings = storage.list_structs(AssistantSettings)
        assert len(settings) == 1


class TestAssistantMultipleAgents:
    @pytest.fixture
    def llm(self):
        return LlmConfig(model="gpt-4", provider="openai")

    @pytest.fixture
    def assistant_settings(self, llm):
        return AssistantSettings(llm=llm)

    @pytest.fixture
    def storage(self):
        return MemoryStorage()

    def test_multiple_agents(self, storage, assistant_settings, llm):
        agent1 = AgentConfig(name="agent1", llm=llm, system_prompt="Agent 1")
        agent2 = AgentConfig(name="agent2", llm=llm, system_prompt="Agent 2")
        
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent1, agent2]
        )
        
        agents = assistant.available_agents
        assert "agent1" in agents
        assert "agent2" in agents

    def test_switch_between_agents(self, storage, assistant_settings, llm):
        agent1 = AgentConfig(name="agent1", llm=llm, system_prompt="Agent 1")
        agent2 = AgentConfig(name="agent2", llm=llm, system_prompt="Agent 2")
        
        assistant = Assistant(
            storage=storage,
            default_settings=assistant_settings,
            agents=[agent1, agent2]
        )
        
        assistant.change_agent("agent1")
        assert assistant.current_agent_name == "agent1"
        
        assistant.change_agent("agent2")
        assert assistant.current_agent_name == "agent2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])