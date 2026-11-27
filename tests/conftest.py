import pytest
from unittest.mock import MagicMock
from aideforge.abc.storage import AbstractStorage
from aideforge.abc.agent_runtime import AbstractAgentRuntime
from aideforge.abc.agent_blueprint import AbstractAgentBlueprint
from aideforge.structs.session import AgentState
from aideforge.utils import create_id
from aideforge.agent.runtime import _models_in_use


pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_terminal,
    pytest.mark.requires_llm,
]


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear global model cache between tests for parallel execution."""
    yield
    _models_in_use.clear()


@pytest.fixture
def mock_storage():
    return MagicMock(spec=AbstractStorage)


@pytest.fixture
def mock_agent_runtime():
    runtime = MagicMock(spec=AbstractAgentRuntime)
    runtime.name = "test_agent"
    runtime.agent_id = create_id("agent")
    runtime.state = MagicMock(spec=AgentState)
    return runtime


@pytest.fixture
def mock_agent_blueprint():
    bp = MagicMock(spec=AbstractAgentBlueprint)
    bp.name = "test_agent"
    bp.instantiate.return_value = mock_agent_runtime
    return bp


@pytest.fixture
def mock_llm_config():
    """Mock LLM config for integration tests - avoids real LLM calls."""
    from aideforge.structs.llm import LlmConfig
    return LlmConfig(model="test-model", provider="mock", parameters={})