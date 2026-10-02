__all__ = ["ExitCommand"]

from . import BaseCommand, CommandResult, ExitAssistant


class ExitCommand(BaseCommand):
    name: str = "exit"
    description: str = "Exit assistant"

    def execute(self, args) -> CommandResult:
        # TODO: delete assistant's agents?
        return ExitAssistant()
