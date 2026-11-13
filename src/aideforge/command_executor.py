from aideforge.abc.assistant import AbstractAssistant
from aideforge.abc.command import BaseCommand
from aideforge.results import (
    CommandExecutionError,
    CommandResult,
    InternalError,
    InvalidArgument,
    UnknownCommand,
)

__all__ = [
    "CommandExecutor",
]


class CommandExecutor:
    """Registry and dispatcher for :class:`BaseCommand` instances.

    Commands are registered by class and instantiated with a reference to
    the :class:`AbstractAssistant`. The executor dispatches slash-commands
    by looking up registered command names.

    Parameters
    ----------
    assistant : AbstractAssistant
        The assistant instance passed to each command during instantiation.

    Attributes
    ----------
    commands : dict[str, BaseCommand]
        A mapping of command names to instantiated command objects.
    """

    def __init__(self, assistant: AbstractAssistant) -> None:
        """Initialize the executor with an empty command registry.

        Parameters
        ----------
        assistant : AbstractAssistant
            The assistant instance passed to each registered command.
        """
        self._commands: dict[str, BaseCommand] = {}
        self._aliases: dict[str, BaseCommand] = {}
        if not isinstance(assistant, AbstractAssistant):
            raise ValueError(
                f"Expected an AbstractAssistant, "
                "got {type(assistant).__name__} instead."
            )
        self._assistant = assistant

    def register_command(self, command: type[BaseCommand]) -> None:
        """Register a command by instantiating and storing it.

        Parameters
        ----------
        command : type[BaseCommand]
            The command to register. Must define a non-empty ``name``
            attribute.

        Raises
        ------
        ValueError
            If the command does not define a valid ``name`` attribute,
            or if a command with that name is already registered.
        """
        if not hasattr(command, "name") or not command.name:
            raise ValueError("Command must define a valid 'name' attribute.")

        if command.name in self._commands:
            raise ValueError(f"Command '{command.name}' is already registered.")

        instance = command(self._assistant)
        self._commands[command.name] = instance
        if command.aliases is not None:
            aliases = command.aliases
            if not isinstance(aliases, list):
                raise ValueError(
                    f"Expected list of aliases, "
                    "got {type(aliases).__name__} instead."
                )
            for alias in aliases:
                self._aliases[alias] = instance

    def execute_command(self, name: str, arg_list: list[str]) -> CommandResult:
        """Find the registered command and execute it with the command line
        argument list

        Parses the provided arguments using the command's parser, then
        invokes the command's execution function.

        Parameters
        ----------
        name : str
            The name (or an alias) of the command to execute.
        arg_list : list[str]
            A list of argument strings to parse and pass to the command.

        Returns
        -------
        CommandResult
            The result of executing the command. Returns an
            :class:`UnknownCommand` error if the command is not registered.
            Returns an :class:`InvalidArgument` error if argument parsing
            fails. Returns a :class:`CommandExecutionError` if the command
            raises an exception during execution.
        """
        command = self._commands.get(name, self._aliases.get(name))
        if command is None:
            return UnknownCommand(error_message=f"Command '{name}' not found.")

        args, err = command.parse_args(arg_list)
        if err:
            return InvalidArgument(error_message=str(err))
        try:
            assert hasattr(
                args, "func"
            ), f"Command '{name}' does not have an execution path."
            return args.func(args)
        except AssertionError as exc:
            return InternalError.from_exception(exc)
        except Exception as exc:
            return CommandExecutionError(error_message=str(exc))

    @property
    def commands(self) -> dict[str, BaseCommand]:
        """Registered commands mapping."""
        return self._commands
