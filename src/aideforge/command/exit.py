from argparse import Namespace

from . import BaseCommand, CommandResult, ExitAssistant

__all__ = ["ExitCommand"]


class ExitCommand(BaseCommand):
    """Command that signals the assistant to exit.

    When executed, returns an :class:`ExitAssistant` result which causes
    the interface loop to terminate.

    Attributes
    ----------
    name : str
        The command name (``"exit"``).
    description : str
        A short description of the command.
    """

    name: str = "exit"
    description: str = "Exit assistant"

    def execute(self, args: Namespace) -> CommandResult:
        """Signal the assistant to exit.

        Parameters
        ----------
        args : Namespace
            Parsed arguments (unused by this command).

        Returns
        -------
        CommandResult
            An :class:`ExitAssistant` instance indicating the assistant
            should terminate.
        """
        return ExitAssistant()
