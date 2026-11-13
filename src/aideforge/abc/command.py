import argparse
from collections.abc import Sequence
from typing import Any

from aideforge.abc.assistant import AbstractAssistant
from aideforge.results import CommandResult, Success

__all__ = [
    "BaseCommand",
    "CommandArgumentParser",
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

    def __init__(self, assistant: AbstractAssistant) -> None:
        """Initialize the command with an argument parser.

        Creates the argument parser using the command's ``name`` and
        ``description``, sets the default execution function, and calls
        :meth:`setup` to allow subclasses to configure arguments.

        Parameters
        ----------
        assistant : AbstractAssistant
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
