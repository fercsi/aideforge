import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from aideforge.agent.types import ActionRequest, ApprovalDecision, AgentSetup
from aideforge.structs.llm import LlmConfig
from aideforge.utils import create_id, IdType
from langchain.messages import SystemMessage
from langchain.tools import BaseTool


class TestActionRequest:
    def test_creation(self):
        req = ActionRequest(
            name="test_tool",
            args={"arg1": "value1"},
            description="Test action"
        )
        assert req.name == "test_tool"
        assert req.args == {"arg1": "value1"}
        assert req.description == "Test action"

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError):
            ActionRequest(name="test")


class TestApprovalDecision:
    def test_values(self):
        assert ApprovalDecision.APPROVE == "approve"
        assert ApprovalDecision.REJECT == "reject"
        assert ApprovalDecision.ALWAYS == "always"
        assert ApprovalDecision.ALWAYS_ALL_TOOLS == "always_all_tools"
        assert ApprovalDecision.APPROVE_FOR_RUN == "approve_for_run"
        assert ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN == "approve_all_tools_for_run"


class MockBaseTool(BaseTool):
    name: str = "mock_tool"
    description: str = "Mock tool"
    
    def _run(self, *args, **kwargs):
        return "mock result"


class TestAgentSetup:
    def test_creation_with_required_fields(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        agent_id = create_id("agent")
        
        setup = AgentSetup(
            agent_id=agent_id,
            name="test_agent",
            llm_config=llm,
            system_message=SystemMessage(content="test")
        )
        assert setup.agent_id == agent_id
        assert setup.name == "test_agent"
        assert setup.llm_config == llm
        assert setup.tools is None
        assert setup.tool_policy is None
        assert setup.interrupt_on is None
        assert setup.threads is None

    def test_creation_with_optional_fields(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        agent_id = create_id("agent")
        tools = [MockBaseTool()]
        tool_policy = {"tool1": "allow"}
        interrupt_on = {"tool1": {"allowed_decisions": ["approve"]}}
        threads = [create_id("thread")]
        
        setup = AgentSetup(
            agent_id=agent_id,
            name="test_agent",
            llm_config=llm,
            tools=tools,
            tool_policy=tool_policy,
            system_message=SystemMessage(content="test"),
            interrupt_on=interrupt_on,
            threads=threads
        )
        assert setup.tools == tools
        assert setup.tool_policy == tool_policy
        assert setup.interrupt_on == interrupt_on
        assert setup.threads == threads

    def test_missing_required_fields_raise(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        with pytest.raises(ValidationError):
            AgentSetup(agent_id=create_id("agent"), name="test", llm_config=llm)