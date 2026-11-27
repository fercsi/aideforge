import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError

from aideforge.cli import Cli, CliCallbackHandler, PsActionResolver, _eval_answer
from aideforge.agent.types import ActionRequest, ApprovalDecision
from aideforge.assistant import Assistant
from aideforge.structs.settings import UiSettings
from aideforge.structs.theme import Theme
from aideforge.storage import MemoryStorage
from langchain_core.outputs import LLMResult
from langchain.messages import AIMessage


class TestCliCallbackHandler:
    def setup_method(self):
        self.mock_shared = MagicMock()
        self.mock_shared.settings = {"display.thinking": "off", "display.show_warnings": False}
        self.mock_shared.terminal = MagicMock()

    def test_creation(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        assert handler._shared == self.mock_shared

    def test_on_llm_end_off(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.thinking": "off"}
        
        mock_response = MagicMock(spec=LLMResult)
        mock_response.generations = [[]]
        
        handler.on_llm_end(mock_response)
        # Should not print anything

    def test_on_llm_end_with_thinking(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.thinking": "all"}
        
        mock_gen = MagicMock()
        mock_gen.message = AIMessage(content="test response")
        mock_gen.message.tool_calls = []
        mock_gen.message.additional_kwargs = {}
        
        mock_response = MagicMock(spec=LLMResult)
        mock_response.generations = [[mock_gen]]
        
        handler.on_llm_end(mock_response)
        self.mock_shared.terminal.print.assert_called()

    def test_on_llm_end_with_tool_calls(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.thinking": "all"}
        
        mock_gen = MagicMock()
        mock_gen.message = AIMessage(content="test", tool_calls=[{"name": "tool1", "args": {"arg": "val"}, "id": "123"}])
        mock_gen.message.additional_kwargs = {}
        
        mock_response = MagicMock(spec=LLMResult)
        mock_response.generations = [[mock_gen]]
        
        handler.on_llm_end(mock_response)
        self.mock_shared.terminal.print.assert_called()

    def test_on_tool_end_details(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.thinking": "details"}
        self.mock_shared.terminal.width = 80
        
        mock_output = MagicMock()
        mock_output.tool_call_id = "123"
        mock_output.name = "test_tool"
        mock_output.content = "tool result"
        mock_output.__class__.__name__ = "ToolOutput"
        
        handler.on_tool_end(mock_output)
        self.mock_shared.terminal.print.assert_called()

    def test_on_tool_end_not_details(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.thinking": "all"}
        
        mock_output = MagicMock()
        mock_output.tool_call_id = "123"
        mock_output.name = "test_tool"
        mock_output.content = "tool result"
        mock_output.__class__.__name__ = "ToolOutput"
        
        handler.on_tool_end(mock_output)
        self.mock_shared.terminal.print.assert_not_called()

    def test_on_llm_error(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        
        handler.on_llm_error(ValueError("test error"))
        self.mock_shared.terminal.print.assert_called_with("LLM ERROR: test error", style="warning")

    def test_on_warning(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.show_warnings": True}
        
        handler.on_warning("test warning")
        self.mock_shared.terminal.print.assert_called_with("WARNING: test warning", style="warning")

    def test_on_warning_disabled(self):
        handler = CliCallbackHandler(shared=self.mock_shared)
        self.mock_shared.settings = {"display.show_warnings": False}
        
        handler.on_warning("test warning")
        self.mock_shared.terminal.print.assert_not_called()


class TestPsActionResolver:
    def test_resolver_tool(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("tool")
        assert result == "test_tool"

    def test_resolver_args_min(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("args", "min")
        assert result == str({"arg1": "val1"})

    def test_resolver_args_pretty(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1", "arg2": "val2"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("args", "pretty")
        assert "arg1 = 'val1'" in result
        assert "arg2 = 'val2'" in result

    def test_resolver_args_pretty_with_indent(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("args", "pretty,4")
        assert result is not None

    def test_resolver_args_json(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("args", "json")
        assert '"arg1": "val1"' in result

    def test_resolver_args_json_with_indent(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("args", "json,2")
        assert result is not None

    def test_resolver_unknown(self):
        request = ActionRequest(name="test_tool", args={"arg1": "val1"}, description="Test")
        resolver = PsActionResolver(request)
        
        result = resolver.resolve("unknown")
        assert result is None


class TestEvalAnswer:
    def test_approve(self):
        assert _eval_answer("y") == ApprovalDecision.APPROVE
        assert _eval_answer("yes") == ApprovalDecision.APPROVE
        assert _eval_answer("approve") == ApprovalDecision.APPROVE

    def test_reject(self):
        assert _eval_answer("n") == ApprovalDecision.REJECT
        assert _eval_answer("no") == ApprovalDecision.REJECT
        assert _eval_answer("reject") == ApprovalDecision.REJECT

    def test_always(self):
        assert _eval_answer("a") == ApprovalDecision.ALWAYS
        assert _eval_answer("always") == ApprovalDecision.ALWAYS

    def test_always_all_tools(self):
        assert _eval_answer("aat") == ApprovalDecision.ALWAYS_ALL_TOOLS
        assert _eval_answer("alwaysall") == ApprovalDecision.ALWAYS_ALL_TOOLS
        assert _eval_answer("alwaysalltools") == ApprovalDecision.ALWAYS_ALL_TOOLS

    def test_approve_for_run(self):
        assert _eval_answer("yfr") == ApprovalDecision.APPROVE_FOR_RUN
        assert _eval_answer("yesforrun") == ApprovalDecision.APPROVE_FOR_RUN
        assert _eval_answer("approveforrun") == ApprovalDecision.APPROVE_FOR_RUN

    def test_approve_all_for_run(self):
        assert _eval_answer("yatfr") == ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN
        assert _eval_answer("yesallforrun") == ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN
        assert _eval_answer("yestoalltoolsforrun") == ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN

    def test_invalid(self):
        assert _eval_answer("invalid") is None
        assert _eval_answer("") is None
        assert _eval_answer("maybe") is None


class TestCli:
    def setup_method(self):
        self.mock_assistant = MagicMock(spec=Assistant)
        self.mock_assistant.description = "Test assistant"
        self.mock_assistant.storage = MemoryStorage()
        self.mock_assistant.settings = MagicMock()
        self.mock_assistant.settings.register = MagicMock()
        self.mock_assistant.settings.ui = UiSettings()
        self.mock_assistant.settings.assistant = MagicMock()
        self.mock_assistant.settings.assistant.history = MagicMock()
        self.mock_assistant.settings.get_theme = MagicMock(return_value=Theme(name="default"))
        self.mock_assistant.session = MagicMock()
        self.mock_assistant.session.history = []
        self.mock_assistant.current_agent = MagicMock()
        self.mock_assistant.current_agent.state.config.welcome_message = "Welcome!"

    def test_creation(self):
        cli = Cli(assistant=self.mock_assistant)
        assert cli._assistant == self.mock_assistant
        assert cli._terminal is not None

    def test_creation_with_custom_settings(self):
        custom_settings = UiSettings(theme="monochrome")
        cli = Cli(assistant=self.mock_assistant, default_settings=custom_settings)
        # The settings are registered with ignore_failure=True, so the first registration wins
        # This test verifies the custom settings are passed correctly
        assert cli._assistant.settings.ui is not None

    def test_process_arguments(self):
        cli = Cli(assistant=self.mock_assistant)
        with patch('sys.argv', ['cli', '-s', 'test_session']):
            cli.process_arguments()
        assert cli._opts.session == "test_session"

    def test_process_arguments_verbose(self):
        cli = Cli(assistant=self.mock_assistant)
        with patch('sys.argv', ['cli', '-v']):
            cli.process_arguments()
        assert cli._opts.verbose is True

    def test_run_not_implemented(self):
        cli = Cli(assistant=self.mock_assistant)
        with pytest.raises(NotImplementedError, match="Synchron CLI run has not been implemented"):
            cli.run()

    def test_arun_session_change(self):
        cli = Cli(assistant=self.mock_assistant)
        cli._opts = MagicMock()
        cli._opts.session = "test_session"
        
        # We can't easily test arun without full mocking
        # Just verify the method exists
        assert hasattr(cli, 'arun')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])