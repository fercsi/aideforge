from argparse import Namespace

from . import BaseCommand, CommandResult, ExitAssistant

__all__ = ["ExitCommand"]


class ExitCommand(BaseCommand):
    name: str = "exit"
    description: str = "Exit assistant"

    def execute(self, args: Namespace) -> CommandResult:
        return ExitAssistant()
