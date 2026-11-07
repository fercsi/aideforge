import traceback
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field

__all__ = [
    "CommandExecutionError",
    "CommandOutput",
    "Error",
    "ErrorCode",
    "ExitAssistant",
    "InternalError",
    "InvalidArgument",
    "LlmError",
    "LlmResponse",
    "Result",
    "Success",
    "UnknownCommand",
]


class Result(BaseModel):
    """Represents a result returned by prompt processors

    Attributes
    ----------
    status : str
        Type of the result
    """

    status: str


class Success(Result):
    """Represents a successful command result.

    Attributes
    ----------
    status : Literal["ok"]
        The status of the result, always ``"ok"``.
    """

    status: Literal["ok"] = "ok"


# ERRONEAUS RESULTS


class Error(Result):
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


class UnknownCommand(Error):
    """Error raised when an unknown command is requested.

    Attributes
    ----------
    error_code : Literal[ErrorCode.UNKNOWN_COMMAND]
        The specific error code for unknown commands.
    """

    error_code: Literal[ErrorCode.UNKNOWN_COMMAND] = ErrorCode.UNKNOWN_COMMAND


class InvalidArgument(Error):
    """Error raised when invalid arguments are provided to a command.

    Attributes
    ----------
    error_code : Literal[ErrorCode.INVALID_ARGUMENT]
        The specific error code for invalid arguments.
    """

    error_code: Literal[ErrorCode.INVALID_ARGUMENT] = ErrorCode.INVALID_ARGUMENT


class CommandExecutionError(Error):
    """Error raised when a command fails during execution.

    Attributes
    ----------
    error_code : Literal[ErrorCode.COMMAND_EXECUTION_ERROR]
        The specific error code for command execution errors.
    """

    error_code: Literal[
        ErrorCode.COMMAND_EXECUTION_ERROR
    ] = ErrorCode.COMMAND_EXECUTION_ERROR


class LlmError(Error):
    """Error raised when communication with the LLM fails.

    Attributes
    ----------
    error_code : Literal[ErrorCode.LLM_ERROR]
        The specific error code for LLM errors.
    """

    error_code: Literal[ErrorCode.LLM_ERROR] = ErrorCode.LLM_ERROR


class InternalError(Error):
    """Error raised for unexpected internal failures.

    Attributes
    ----------
    error_code : Literal[ErrorCode.INTERNAL_ERROR]
        The specific error code for internal errors.
    type : str
        The exception class name.
    message : str
        The exception message (``str(exc)``).
    args : list[Any]
        The exception arguments.
    traceback : list[str] | None
        The formatted traceback stack lines, if available.
    """

    error_code: Literal[ErrorCode.INTERNAL_ERROR] = ErrorCode.INTERNAL_ERROR
    type: str = Field(..., description="Exception class name")
    message: str = Field(..., description="Exception message (str(e))")
    args: list[Any] = Field(
        default_factory=list, description="Exception arguments"
    )
    traceback: list[str] | None = Field(
        default=None, description="Formatted traceback stack lines"
    )

    @classmethod
    def from_exception(
        cls, exc: BaseException, include_traceback: bool = True
    ) -> Self:
        """Create an :class:`InternalError` instance from a :class:`BaseException`.

        Parameters
        ----------
        exc : BaseException
            The exception to convert.
        include_traceback : bool, optional
            Whether to include the formatted traceback in the result,
            by default ``True``.

        Returns
        -------
        InternalError
            An :class:`InternalError` populated with the exception's
            type, message, args, and (optionally) traceback.
        """
        tb_lines = None
        if include_traceback and exc.__traceback__:
            tb_lines = traceback.format_exception(
                type(exc), exc, exc.__traceback__
            )

        type_name = exc.__class__.__name__
        return cls(
            error_message=f"Internal error ({type_name}): {exc}.",
            type=type_name,
            message=str(exc),
            args=[
                str(arg) if isinstance(arg, BaseException) else arg
                for arg in exc.args
            ],
            traceback=tb_lines,
        )


# COMMAND RELATED RESULTS


class CommandOutput(Result):
    """Result model for command output (non-LLM-response text).

    Attributes
    ----------
    status : Literal["output"]
        The status of the result.
    content : str
        The output text produced by the command.
    """

    status: Literal["output"] = "output"
    content: str


class ExitAssistant(Result):
    """Result model indicating the assistant should exit.

    Attributes
    ----------
    status : Literal["exit"]
        The status of the result, always ``"exit"``.
    """

    status: Literal["exit"] = "exit"


#: CommandSuccess
#: --------------
#:
#: Represents the possible successful outcomes of a command.
#:
#: This type alias can be one of:
#:
#: - ``Success``
#: - ``CommandOutput``
#: - ``ExitAssistant``
CommandSuccess = Annotated[
    Success | CommandOutput | ExitAssistant, Field(discriminator="status")
]

#: CommandError
#: ------------
#:
#: Represents the possible failed outcomes of a command.
#:
#: This type alias can be one of:
#:
#:   - ``UnknownCommand``
#:   - ``InvalidArgument``
#:   - ``CommandExecutionError``
#:   - ``InternalError``
CommandError = Annotated[
    UnknownCommand | InvalidArgument | CommandExecutionError | InternalError,
    Field(discriminator="error_code"),
]

#: CommandResult
#: -------------
#:
#: Represents the possible outcomes of a command.
#:
#: This type alias can be one of:
#:
#: - ``CommandSuccess``
#: - ``CommandError``
CommandResult = Annotated[
    CommandSuccess | CommandError, Field(discriminator="status")
]

# LLM RELATED RESULTS


class LlmResponse(BaseModel):
    """Represents a successful LLM response.

    Attributes
    ----------
    status : Literal["response"]
        The status of the response, always ``"response"``.
    content : str or None
        The text content returned by the LLM.
    """

    status: Literal["response"] = "response"
    content: str | None

# ASSISTANT RELATED RESULTS

PromptError = Annotated[
    CommandError | LlmError, Field(descriminator="error_code")
]

PromptResult = Annotated[
    LlmResponse | CommandSuccess | PromptError, Field(descriminator="status")
]

