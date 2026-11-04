from __future__ import annotations

import argparse
import importlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from aideforge.status import (
    UnknownCommand,
    InternalError,
    InvalidArgument,
    CommandExecutionError,
    Success,
)

if TYPE_CHECKING:
    from aideforge.assistant import Assistant

__all__ = [
    "BaseCommand",
    "CommandError",
    "CommandExecutor",
    "CommandOutput",
    "CommandResult",
    "CommandSuccess",
    "ExitAssistant",
    "default_commands",
]


class ExitAssistant(BaseModel):
    """Result model indicating the assistant should exit.

    Attributes
    ----------
    status : Literal["exit"]
        The status of the result, always ``"exit"``.
    """

    status: Literal["exit"] = "exit"


class CommandOutput(BaseModel):
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


class CommandArgumentError(BaseException):
    """Exception raised when command argument parsing fails.

    Parameters
    ----------
    message : str
        A description of the parsing error.

    Attributes
    ----------
    message : str
        The error message describing the argument parsing failure.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception with an error message.

        Parameters
        ----------
        message : str
            A description of the parsing error.
        """
        super().__init__(message)
        self.message = message


class CommandArgumentParser(argparse.ArgumentParser):
    """Custom argument parser serving builtin and customer commands

    Overrides :meth:`error` to raise a :class:`CommandArgumentError`
    instead of calling ``sys.exit``. Also supresses argparse default
    ``--help`` option (``/help`` command is used in assistant interface
    instead) and sets common defaults for subparser configuration
    (see :meth:`add_subparser`).

    Parameters
    ----------
    *args : Any
        Positional arguments passed to :class:`argparse.ArgumentParser`.
    **kwargs : Any
        Keyword arguments passed to :class:`argparse.ArgumentParser`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parser with default subparser settings.

        Sets ``add_help`` to ``False`` by default if not provided.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the parent constructor.
        **kwargs : Any
            Keyword arguments forwarded to the parent constructor.
        """
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)

    def add_subparsers(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add subparsers with sensible default settings.

        Sets ``required`` to ``True``, ``metavar`` to ``"SUBCOMMAND"``,
        and ``help`` to ``"%(choices)s"`` by default if not provided. E.g.::

          usage: /agent SUBCOMMAND ...
          ...
          positional arguments:
            SUBCOMMAND  list, use, swap
          ...

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the parent method.
        **kwargs : Any
            Keyword arguments forwarded to the parent method.

        Returns
        -------
        argparse.Action
            The subparsers action object.
        """
        kwargs.setdefault("required", True)
        kwargs.setdefault("metavar", "SUBCOMMAND")
        kwargs.setdefault("help", "%(choices)s")
        return super().add_subparsers(*args, **kwargs)

    def error(self, message: str) -> None:
        """Handle a parsing error by raising an exception.

        Overrides the default behavior of printing usage and exiting.

        Parameters
        ----------
        message : str
            The error message describing the parsing failure.

        Raises
        ------
        CommandArgumentError
            Always raised with *message* as the argument.
        """
        raise CommandArgumentError(message)


class BaseCommand:
    """Base class for implementing slash-commands.

    Subclasses must define the ``name`` and ``description`` class
    attributes, and may override :meth:`setup` to configure the
    argument parser and :meth:`execute` to implement the command logic.
    A separate method can be assigned to each subcommand.

    Attributes
    ----------
    name : str
        The command name, used as the slash-command identifier.
    aliases : list[str] | None, optional
        Aliases that can be used instead of ``name``. Note that, unlike
        ``name``, aliases are weak, meaning that they can overwrite each
        other or be overwritten by any command.
    description : str
        A short description shown in help output.
    category : str, optional
        A category used to group commands in ``/help`` when the number of
        commands exceeds seven. Defaults to ``"misc"``.
    parser : CommandArgumentParser
        The argument parser for this command.
    subparsers : dict[str, CommandArgumentParser]
        A dictionary of sub-parsers for sub-commands.

    Examples
    --------
    >>> class TestCommand(BaseCommand):
    ...     name = "test"
    ...     description = "This is just a test command"
    ...
    ...     def setup(self, parser, subparsers, **kwargs):
    ...         parser.add_argument(
    ...             "-v", "--verbose", action='store_true'
    ...         )
    ...         parser_group = parser.add_subparsers(required=False)
    ...
    ...         subparser = parser_group.add_parser(
    ...             "subcmd1", description="Subcommand #1"
    ...         )
    ...         subparser.set_defaults(func=self.subcmd1)
    ...         subparsers["subcmd1"] = subparser
    ...
    ...         subparser = parser_group.add_parser(
    ...             "subcmd2", description="Subcommand #2"
    ...         )
    ...         subparser.add_argument(
    ...             "name", metavar="NAME", help="An extra argument to subcommand #2"
    ...         )
    ...         subparser.set_defaults(func=self.subcmd2)
    ...         subparsers["subcmd2"] = subparser
    ...
    ...     def subcmd1(self, args):
    ...         ...
    ...     def subcmd2(self, args):
    ...         # you can use args.name here
    ...         ...
    ...     def execute(self, args):
    ...         # This is executed when the command has no subcommands, or
    ...         # when it does have subcommands but the caller invokes it
    ...         # without specifying one (and this option is enabled in
    ...         # ``parser_group``).
    ...         #
    ...         # --version can be used in each command
    ...         ...
    
    Running assistant with the new (``test``) command:

    .. code-block:: console

        $ ./assistant
        ...
        CLI> /test -v subcmd2 "my name"
        ...
    """

    name: str
    aliases: list[str] | None = None
    description: str
    category: str = "misc"

    def __init__(self, assistant: Assistant) -> None:
        """Initialize the command with an argument parser.

        Creates the argument parser using the command's ``name`` and
        ``description``, sets the default execution function, and calls
        :meth:`setup` to allow subclasses to configure arguments.

        Parameters
        ----------
        assistant : Assistant
            The assistant instance this command is registered with.
        """
        self._assistant = assistant
        self.parser = CommandArgumentParser(
            prog="/" + self.name,
            description=self.description,
        )
        self.parser.set_defaults(func=self.execute)
        self.subparsers = {}
        self.setup(parser=self.parser, subparsers=self.subparsers)

    def parse_args(
        self, args: Sequence[str]
    ) -> tuple[argparse.Namespace | None, CommandArgumentError | None]:
        """Parse command arguments.

        Parameters
        ----------
        args : Sequence[str]
            The list of argument strings to parse.

        Returns
        -------
        tuple[argparse.Namespace | None, CommandArgumentError | None]
            A tuple of ``(parsed_args, error)``. If parsing succeeds,
            *parsed_args* contains the namespace and *error* is ``None``.
            If parsing fails, *parsed_args* is ``None`` and *error*
            contains the :class:`CommandArgumentError`.
        """
        try:
            args = self.parser.parse_args(args)
        except CommandArgumentError as exc:
            return None, exc
        return args, None

    def setup(
        self,
        parser: CommandArgumentParser,
        subparsers: dict[str, CommandArgumentParser],
        **kwargs: Any,
    ) -> None:
        """Configure the argument parser for this command.

        Override this method in subclasses to add arguments and
        sub-parsers. An example of how to specify the method can be found
        in the documentation of the :class:`BaseCommand` class.

        The parameters of the ``setup`` method are planned to be extended,
        so the use of the ``**kwargs`` parameter is recommended for future
        compatibility.

        Parameters
        ----------
        parser : CommandArgumentParser
            The main argument parser for this command.
        subparsers : dict[str, CommandArgumentParser]
            A dictionary to populate with sub-parsers.
        kwargs : Any
            Use this for future compatibility.
        """
        pass

    def execute(self, args: argparse.Namespace) -> CommandResult:
        """Execute the command.

        Override this method in subclasses to implement command logic.

        Parameters
        ----------
        args : argparse.Namespace
            The parsed command-line arguments.

        Returns
        -------
        CommandResult
            The result of the command.
        """
        return Success()


class CommandExecutor:
    """Registry and dispatcher for :class:`BaseCommand` instances.

    Commands are registered by class and instantiated with a reference
    to the :class:`Assistant`. The executor dispatches slash-commands by
    looking up registered command names.

    Parameters
    ----------
    assistant : Assistant
        The assistant instance passed to each command during instantiation.

    Attributes
    ----------
    commands : dict[str, BaseCommand]
        A mapping of command names to instantiated command objects.
    """

    def __init__(self, assistant: Assistant) -> None:
        """Initialize the executor with an empty command registry.

        Parameters
        ----------
        assistant : Assistant
            The assistant instance passed to each registered command.
        """
        from aideforge.assistant import Assistant
        self._commands: dict[str, BaseCommand] = {}
        self._aliases: dict[str, BaseCommand] = {}
        if not isinstance(assistant, Assistant):
            raise ValueError(f"Expected an Assistant, got {type(assistant).__name__} instead.")
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
                raise ValueError(f"Expected list of aliases, got {type(aliases).__name__} instead.")
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
            assert hasattr(args, "func"), f"Command '{name}' does not have an execution path."
            return args.func(args)
        except AssertionError as exc:
            return InternalError.from_exception(exc)
        except Exception as exc:
            return CommandExecutionError(error_message=str(exc))

    @property
    def commands(self) -> dict[str, BaseCommand]:
        return self._commands

def _import_commands() -> list[type[BaseCommand]]:
    """Dynamically import and discover :class:`BaseCommand` subclasses.

    Scans the current package directory for ``*.py`` files (excluding
    ``__init__.py``), imports each module, and collects all classes that
    are subclasses of :class:`BaseCommand` defined in that module.

    Returns
    -------
    list[type[BaseCommand]]
        A list of discovered command classes.
    """
    commands = []

    current_file = Path(__file__).resolve()
    package_dir = current_file.parent
    package_name = __package__ if __package__ else package_dir.name

    for file_path in package_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        # In case of some analytics tools, no package_name is present
        module_name = (
            f"{package_name}.{file_path.stem}"
            if package_name
            else file_path.stem
        )

        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseCommand)
                and obj is not BaseCommand
                and obj.__module__ == module.__name__
            ):
                commands.append(obj)
    return commands


default_commands = _import_commands()
