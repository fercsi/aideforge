from enum import StrEnum
from typing import Literal, Annotated
from pydantic import BaseModel, Field


class Success(BaseModel):
    status: Literal["ok"] = "ok"


class BaseError(BaseModel):
    status: Literal["error"] = "error"
    error_message: str


class ErrorCode(StrEnum):
    UNKNOWN_COMMAND = "unknown_command"
    INVALID_ARGUMENT = "invalid_argument"
    COMMAND_EXECUTION_ERROR = "command_execution_error"
    LLM_ERROR = "llm_error"
    INTERNAL_ERROR = "internal_error"


class UnknownCommand(BaseError):
    error_code: Literal[ErrorCode.UNKNOWN_COMMAND] = ErrorCode.UNKNOWN_COMMAND


class InvalidArgument(BaseError):
    error_code: Literal[ErrorCode.INVALID_ARGUMENT] = ErrorCode.INVALID_ARGUMENT


class CommandExecutionError(BaseError):
    error_code: Literal[
        ErrorCode.COMMAND_EXECUTION_ERROR
    ] = ErrorCode.COMMAND_EXECUTION_ERROR


class LlmError(BaseError):
    error_code: Literal[ErrorCode.LLM_ERROR] = ErrorCode.LLM_ERROR


class InternalError(BaseError):
    error_code: Literal[ErrorCode.INTERNAL_ERROR] = ErrorCode.INTERNAL_ERROR
    exception_details: str


# AnyError = Annotated[
#    UnknownCommand
#    | InvalidArgument
#    | CommandExecutionError
#    | LlmError
#    | InternalError,
#    Field(descriminator="error_code")
# ]
#
# PromptStatus = Annotated[Success | AnyError, Field(descriminator="status")]
