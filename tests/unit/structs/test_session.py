import pytest
from pydantic import ValidationError

from aideforge.structs.session import AgentState, SessionState
from aideforge.structs.agent import AgentConfig, ContextInfo
from aideforge.structs.llm import LlmConfig
from aideforge.utils import create_id, IdType


class TestAgentState:
    def test_creation(self):
        agent_config = AgentConfig(name="test_agent")
        llm_config = LlmConfig(model="gpt-4", provider="openai")
        threads = [create_id("thread") for _ in range(2)]

        state = AgentState(
            config=agent_config,
            threads=threads,
            llm=llm_config
        )
        assert state.config.name == "test_agent"
        assert len(state.threads) == 2
        assert state.llm.model == "gpt-4"
        assert isinstance(state.context_info, ContextInfo)
        assert state.tool_policy is None
        assert state.env is None

    def test_with_optional_fields(self):
        agent_config = AgentConfig(name="test_agent")
        llm_config = LlmConfig(model="gpt-4", provider="openai")
        threads = [create_id("thread")]
        tool_policy = {"tool1": "allow", "tool2": "deny"}
        env = {"custom_key": "custom_value"}

        state = AgentState(
            config=agent_config,
            threads=threads,
            llm=llm_config,
            tool_policy=tool_policy,
            env=env
        )
        assert state.tool_policy == tool_policy
        assert state.env == env

    def test_missing_config_raises(self):
        llm_config = LlmConfig(model="gpt-4", provider="openai")
        threads = [create_id("thread")]
        with pytest.raises(ValidationError):
            AgentState(threads=threads, llm=llm_config)

    def test_missing_threads_raises(self):
        agent_config = AgentConfig(name="test_agent")
        llm_config = LlmConfig(model="gpt-4", provider="openai")
        with pytest.raises(ValidationError):
            AgentState(config=agent_config, llm=llm_config)

    def test_missing_llm_raises(self):
        agent_config = AgentConfig(name="test_agent")
        threads = [create_id("thread")]
        with pytest.raises(ValidationError):
            AgentState(config=agent_config, threads=threads)


class TestSessionState:
    def test_default_values(self):
        session = SessionState()
        assert session.kind == "session"
        assert session.version == 1
        assert session.session_id is not None
        assert session.session_id.version == 7
        assert session.agents == {}
        assert session.current_agent is None
        assert session.history == []
        assert session.input_count == 0
        assert session.env is None

    def test_custom_session_id(self):
        custom_id = create_id("session")
        session = SessionState(session_id=custom_id)
        assert session.session_id == custom_id

    def test_with_agents(self):
        agent_config = AgentConfig(name="agent1")
        llm_config = LlmConfig(model="gpt-4", provider="openai")
        agent_id = create_id("agent")
        agent_state = AgentState(config=agent_config, threads=[create_id("thread")], llm=llm_config)

        session = SessionState(
            agents={agent_id: agent_state},
            current_agent=agent_id
        )
        assert agent_id in session.agents
        assert session.current_agent == agent_id

    def test_with_history(self):
        session = SessionState(
            history=["prompt1", "prompt2"],
            input_count=2
        )
        assert session.history == ["prompt1", "prompt2"]
        assert session.input_count == 2

    def test_metadata(self):
        assert SessionState.scope == "user"
        assert SessionState.storage_name == "{struct.session_id}"
        assert SessionState.kinds == "sessions"

    def test_with_env(self):
        session = SessionState(env={"cwd": "/home/user"})
        assert session.env == {"cwd": "/home/user"}