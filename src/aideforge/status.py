from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Success(BaseModel):
    """Represents a successful command result.

    Attributes
    ----------
    status : Literal["ok"]
        The status of the result, always ``"ok"``.
    """

    status: Literal["ok"] = "ok"


class BaseError(BaseModel):
    """Base model for error results.

    Attributes
    ----------
    status : Literal["error"]
        The status of the result, always ``"error"``.
    error_message : str
        A human-readable description of the error.
    """

    status: Literal["error"] = "error"
    error_message: str


class ErrorCode(StrEnum):
    """Enumeration of error codes.

    Attributes
    ----------
    UNKNOWN_COMMAND : str
        The command was not recognised.
    INVALID_ARGUMENT : str
        The provided arguments were invalid.
    COMMAND_EXECUTION_ERROR : str
        The command failed during execution.
    LLM_ERROR : str
        An error occurred while communicating with the LLM.
    INTERNAL_ERROR : str
        An unexpected internal error occurred.
    """

    UNKNOWN_COMMAND = "unknown_command"
    INVALID_ARGUMENT = "invalid_argument"
    COMMAND_EXECUTION_ERROR = "command_execution_error"
    LLM_ERROR = "llm_error"
    INTERNAL_ERROR = "internal_error"


class UnknownCommand(BaseError):
    """Error raised when an unknown command is requested.

    Attributes
    ----------
    error_code : Literal[ErrorCode.UNKNOWN_COMMAND]
        The specific error code for unknown commands.
    """

    error_code: Literal[ErrorCode.UNKNOWN_COMMAND] = ErrorCode.UNKNOWN_COMMAND


class InvalidArgument(BaseError):
    """Error raised when invalid arguments are provided to a command.

    Attributes
    ----------
    error_code : Literal[ErrorCode.INVALID_ARGUMENT]
        The specific error code for invalid arguments.
    """

    error_code: Literal[ErrorCode.INVALID_ARGUMENT] = ErrorCode.INVALID_ARGUMENT


class CommandExecutionError(BaseError):
    """Error raised when a command fails during execution.

    Attributes
    ----------
    error_code : Literal[ErrorCode.COMMAND_EXECUTION_ERROR]
        The specific error code for command execution errors.
    """

    error_code: Literal[
        ErrorCode.COMMAND_EXECUTION_ERROR
    ] = ErrorCode.COMMAND_EXECUTION_ERROR


class LlmError(BaseError):
    """Error raised when communication with the LLM fails.

    Attributes
    ----------
    error_code : Literal[ErrorCode.LLM_ERROR]
        The specific error code for LLM errors.
    """

    error_code: Literal[ErrorCode.LLM_ERROR] = ErrorCode.LLM_ERROR


class InternalError(BaseError):
    """Error raised for unexpected internal failures.

    Attributes
    ----------
    error_code : Literal[ErrorCode.INTERNAL_ERROR]
        The specific error code for internal errors.
    exception_details : str
        Additional details about the exception, including stack trace
        information.
    """

    error_code: Literal[ErrorCode.INTERNAL_ERROR] = ErrorCode.INTERNAL_ERROR
    exception_details: str
