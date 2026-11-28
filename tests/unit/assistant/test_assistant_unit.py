import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pydantic import ValidationError

from aideforge.assistant import Assistant
from aideforge.structs.agent import AgentConfig
from aideforge.structs.llm import LlmConfig
from aideforge.structs.session import SessionState
from aideforge.structs.settings import AssistantSettings, UiSettings
from aideforge.storage import MemoryStorage
from aideforge.tools import ToolPolicy
from aideforge.utils import create_id, IdType
from aideforge.abc.storage import AbstractStorage


class TestAssistant:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.agent_config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Test prompt"
        )
        # Default assistant settings with required LLM
        self.default_settings = AssistantSettings(llm=self.llm)

    def test_creation_with_defaults(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assert assistant._storage == self.mock_storage
        assert assistant._settings is not None
        assert assistant.agent_blueprints is not None

    def test_creation_with_custom_storage(self):
        custom_storage = MagicMock(spec=AbstractStorage)
        assistant = Assistant(storage=custom_storage, default_settings=self.default_settings)
        assert assistant._storage == custom_storage

    def test_creation_with_llm(self):
        assistant = Assistant(storage=self.mock_storage, llm=self.llm, default_settings=self.default_settings)
        assert assistant._current_llm == self.llm

    def test_creation_with_session_dict(self):
        session_data = {"session_id": create_id("session"), "agents": {}}
        assistant = Assistant(storage=self.mock_storage, session=session_data, default_settings=self.default_settings)
        assert assistant._session is not None

    def test_creation_with_session_string(self):
        session_id = create_id("session")
        self.mock_storage.retrieve_struct.return_value = SessionState(session_id=session_id)
        assistant = Assistant(storage=self.mock_storage, session=str(session_id), default_settings=self.default_settings)
        assert assistant._session.session_id == session_id

    def test_creation_with_agents(self):
        agents = [self.agent_config]
        assistant = Assistant(storage=self.mock_storage, agents=agents, default_settings=self.default_settings)
        assert "test_agent" in assistant.agent_blueprints

    def test_prepare(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.prepare()
        self.mock_storage.prepare.assert_called_once()

    def test_cleanup(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.cleanup()
        self.mock_storage.store_struct.assert_called()
        self.mock_storage.cleanup.assert_called_once()

    def test_change_session_none(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.change_session(None)
        assert assistant._session is not None

    def test_change_session_string(self):
        session_id = create_id("session")
        self.mock_storage.retrieve_struct.return_value = SessionState(session_id=session_id)
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.change_session(str(session_id))
        assert assistant._session.session_id == session_id

    def test_available_agents(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        agents = assistant.available_agents
        assert "test_agent" in agents

    def test_change_agent(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        assistant.change_agent("test_agent")
        assert assistant.current_agent_name == "test_agent"

    def test_change_agent_invalid_raises(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.change_agent("nonexistent")
        # The error is raised when trying to use the agent, not when changing
        with pytest.raises(KeyError):
            assistant._get_current_agent()

    def test_register_commands(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        from aideforge.abc.command import BaseCommand
        from aideforge.results import CommandResult
        
        class MockCommand(BaseCommand):
            name = "mock"
            aliases = []
            description = "Mock command"
            
            def execute(self, args: list[str]) -> CommandResult:
                from aideforge.results import Success
                return Success()
        
        assistant.register_commands([MockCommand])
        assert "mock" in assistant.commands

    def test_render_template_basic(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        result = assistant.render_template("Hello {agent}")
        assert "test_agent" in result or "default" in result

    def test_render_template_unknown_raises(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        with pytest.raises(ValueError, match="Unknown template variable"):
            assistant.render_template("Hello {unknown}")

    def test_settings_property(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assert assistant.settings is not None

    def test_storage_property(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assert assistant.storage == self.mock_storage

    def test_description_property(self):
        assistant = Assistant(storage=self.mock_storage, description="Custom", default_settings=self.default_settings)
        assert assistant.description == "Custom"


class TestAssistantTemplateResolution:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.agent_config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Test prompt"
        )
        self.default_settings = AssistantSettings(llm=self.llm)

    def test_resolve_session_id(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        result = assistant._resolve_session("session_id", "")
        assert result == str(assistant._session.session_id)

    def test_resolve_sn(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        result = assistant._resolve_session("sn", "")
        assert result == str(assistant._session.input_count + 1)

    def test_resolve_agent_name(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        assistant.current_agent_name = "test_agent"
        result = assistant._resolve_agent("agent", "")
        assert result == "test_agent"

    def test_resolve_model(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        assistant.current_agent_name = "test_agent"
        with patch.object(assistant, '_get_current_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.state.llm.model = "gpt-4"
            mock_agent.state.llm.provider = "openai"
            mock_get_agent.return_value = mock_agent
            result = assistant._resolve_agent("model", "")
            assert result == "gpt-4"

    def test_resolve_model_provider(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        assistant.current_agent_name = "test_agent"
        with patch.object(assistant, '_get_current_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.state.llm.model = "gpt-4"
            mock_agent.state.llm.provider = "openai"
            mock_get_agent.return_value = mock_agent
            result = assistant._resolve_agent("model_provider", "")
            assert result == "openai"


class TestAssistantSessionManagement:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.default_settings = AssistantSettings(llm=self.llm)

    def test_store_session(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        # Reset mock to ignore calls during initialization
        self.mock_storage.reset_mock()
        assistant._session = SessionState(session_id=create_id("session"), input_count=1)
        assistant._stored_input_count = 0
        assistant._store_session()
        self.mock_storage.store_struct.assert_called_once()

    def test_store_session_no_session(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        self.mock_storage.reset_mock()
        assistant._session = None
        assistant._store_session()
        self.mock_storage.store_struct.assert_not_called()


class TestAssistantAgentManagement:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.agent_config = AgentConfig(
            name="test_agent",
            llm=self.llm,
            system_prompt="Test prompt"
        )
        self.default_settings = AssistantSettings(llm=self.llm)

    def test_get_current_agent_new(self):
        assistant = Assistant(storage=self.mock_storage, agents=[self.agent_config], default_settings=self.default_settings)
        assistant.current_agent_name = "test_agent"
        
        with patch.object(assistant.agent_blueprint, 'instantiate') as mock_instantiate:
            mock_agent = MagicMock()
            mock_agent.agent_id = create_id("agent")
            mock_agent.state.llm = self.llm
            mock_instantiate.return_value = mock_agent
            
            agent = assistant._get_current_agent()
            
            assert agent == mock_agent
            mock_instantiate.assert_called_once()

    def test_get_current_agent_not_found_raises(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        assistant.current_agent_name = "nonexistent"
        
        with pytest.raises(KeyError, match="Agent nonexistent not found"):
            assistant._get_current_agent()


class TestAssistantCommandProcessing:
    def setup_method(self):
        self.mock_storage = MagicMock(spec=AbstractStorage)
        self.llm = LlmConfig(model="gpt-4", provider="openai")
        self.default_settings = AssistantSettings(llm=self.llm)

    def test_process_command_unknown(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        result = assistant._process_command("/unknown arg1 arg2")
        # Should return a PromptError for unknown command
        assert hasattr(result, 'status')
        assert result.status == "error"

    def test_process_command_empty(self):
        assistant = Assistant(storage=self.mock_storage, default_settings=self.default_settings)
        result = assistant._process_command("/")
        assert hasattr(result, 'status')
        assert result.status == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])