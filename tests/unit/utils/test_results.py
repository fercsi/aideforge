import pytest
from aideforge.results import (
    Success,
    Error,
    ErrorCode,
    UnknownCommand,
    InvalidArgument,
    CommandExecutionError,
    LlmError,
    InternalError,
    CommandOutput,
    ExitAssistant,
    LlmResponse,
    CommandSuccess,
    CommandError,
    CommandResult,
    PromptError,
    PromptResult,
)


class TestSuccess:
    def test_default_status(self):
        s = Success()
        assert s.status == "ok"

    def test_explicit_status(self):
        s = Success(status="ok")
        assert s.status == "ok"

    def test_model_dump(self):
        s = Success()
        assert s.model_dump() == {"status": "ok"}


class TestError:
    def test_default_status(self):
        e = Error(error_message="test error")
        assert e.status == "error"
        assert e.error_message == "test error"

    def test_error_message_required(self):
        with pytest.raises(Exception):
            Error()


class TestErrorCode:
    def test_values(self):
        assert ErrorCode.UNKNOWN_COMMAND == "unknown_command"
        assert ErrorCode.INVALID_ARGUMENT == "invalid_argument"
        assert ErrorCode.COMMAND_EXECUTION_ERROR == "command_execution_error"
        assert ErrorCode.LLM_ERROR == "llm_error"
        assert ErrorCode.INTERNAL_ERROR == "internal_error"


class TestUnknownCommand:
    def test_default_error_code(self):
        e = UnknownCommand(error_message="unknown")
        assert e.error_code == ErrorCode.UNKNOWN_COMMAND
        assert e.status == "error"

    def test_model_dump(self):
        e = UnknownCommand(error_message="cmd not found")
        assert e.model_dump() == {
            "status": "error",
            "error_message": "cmd not found",
            "error_code": "unknown_command"
        }


class TestInvalidArgument:
    def test_default_error_code(self):
        e = InvalidArgument(error_message="bad arg")
        assert e.error_code == ErrorCode.INVALID_ARGUMENT


class TestCommandExecutionError:
    def test_default_error_code(self):
        e = CommandExecutionError(error_message="exec failed")
        assert e.error_code == ErrorCode.COMMAND_EXECUTION_ERROR


class TestLlmError:
    def test_default_error_code(self):
        e = LlmError(error_message="llm failed")
        assert e.error_code == ErrorCode.LLM_ERROR


class TestInternalError:
    def test_default_error_code(self):
        e = InternalError(
            error_message="internal",
            type="ValueError",
            message="bad value",
            args=[]
        )
        assert e.error_code == ErrorCode.INTERNAL_ERROR

    def test_from_exception_basic(self):
        try:
            raise ValueError("invalid value")
        except ValueError as exc:
            ie = InternalError.from_exception(exc)
        assert ie.type == "ValueError"
        assert ie.message == "invalid value"
        assert ie.args == ["invalid value"]
        assert ie.traceback is not None
        assert "ValueError: invalid value" in "".join(ie.traceback)

    def test_from_exception_without_traceback(self):
        exc = ValueError("invalid value")
        ie = InternalError.from_exception(exc, include_traceback=False)
        assert ie.traceback is None

    def test_from_exception_with_args(self):
        exc = Exception("arg1", "arg2", 123)
        ie = InternalError.from_exception(exc)
        assert ie.args == ["arg1", "arg2", 123]

    def test_from_exception_nested_exception_in_args(self):
        inner = ValueError("inner")
        outer = Exception("outer", inner)
        ie = InternalError.from_exception(outer)
        assert ie.args == ["outer", "inner"]

    def test_error_message_format(self):
        exc = ValueError("test")
        ie = InternalError.from_exception(exc)
        assert ie.error_message == "Internal error (ValueError): test."


class TestCommandOutput:
    def test_default_status(self):
        co = CommandOutput(content="output text")
        assert co.status == "output"
        assert co.content == "output text"

    def test_model_dump(self):
        co = CommandOutput(content="hello")
        assert co.model_dump() == {"status": "output", "content": "hello"}


class TestExitAssistant:
    def test_default_status(self):
        ea = ExitAssistant()
        assert ea.status == "exit"

    def test_model_dump(self):
        ea = ExitAssistant()
        assert ea.model_dump() == {"status": "exit"}


class TestLlmResponse:
    def test_default_status(self):
        lr = LlmResponse(content="response text")
        assert lr.status == "response"
        assert lr.content == "response text"

    def test_none_content(self):
        lr = LlmResponse(content=None)
        assert lr.content is None


class TestDiscriminatedUnions:
    def test_command_success_success(self):
        cs: CommandSuccess = Success()
        assert cs.status == "ok"

    def test_command_success_command_output(self):
        cs: CommandSuccess = CommandOutput(content="out")
        assert cs.status == "output"

    def test_command_success_exit_assistant(self):
        cs: CommandSuccess = ExitAssistant()
        assert cs.status == "exit"

    def test_command_error_unknown_command(self):
        ce: CommandError = UnknownCommand(error_message="unknown")
        assert ce.error_code == "unknown_command"

    def test_command_error_invalid_argument(self):
        ce: CommandError = InvalidArgument(error_message="bad")
        assert ce.error_code == "invalid_argument"

    def test_command_error_command_execution(self):
        ce: CommandError = CommandExecutionError(error_message="exec")
        assert ce.error_code == "command_execution_error"

    def test_command_error_internal_error(self):
        ce: CommandError = InternalError(
            error_message="internal",
            type="Error",
            message="msg",
            args=[]
        )
        assert ce.error_code == "internal_error"

    def test_command_result_success(self):
        cr: CommandResult = Success()
        assert cr.status == "ok"

    def test_command_result_error(self):
        cr: CommandResult = UnknownCommand(error_message="unknown")
        assert cr.status == "error"

    def test_prompt_error_command_error(self):
        pe: PromptError = UnknownCommand(error_message="unknown")
        assert pe.error_code == "unknown_command"

    def test_prompt_error_llm_error(self):
        pe: PromptError = LlmError(error_message="llm fail")
        assert pe.error_code == "llm_error"

    def test_prompt_result_llm_response(self):
        pr: PromptResult = LlmResponse(content="resp")
        assert pr.status == "response"

    def test_prompt_result_command_success(self):
        pr: PromptResult = Success()
        assert pr.status == "ok"

    def test_prompt_result_prompt_error(self):
        pr: PromptResult = UnknownCommand(error_message="unknown")
        assert pr.status == "error"