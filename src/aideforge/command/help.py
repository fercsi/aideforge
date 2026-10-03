from argparse import Namespace

from . import (
    BaseCommand,
    CommandArgumentParser,
    CommandExecutionError,
    CommandOutput,
    CommandResult,
    InvalidArgument,
    Success,
)

__all__ = ["AgentCommand"]

# TODO: make help text "editable"
_HELP_TEXT = """
This is your AI assistant. You can ask something from the LLM, or you may run
some commands. Commands always start with a "/" character.

Your assistant understands the following commands:

{commands}

Use "/help COMMAND" to get more details.
"""


class HelpCommand(BaseCommand):
    name: str = "help"
    description: str = (
        "Prints general help information or information about commands"
    )

    def setup(self, parser: CommandArgumentParser, **kwargs) -> None:
        parser.add_argument(
            "command_name",
            nargs="?",
            metavar="COMMAND_NAME",
            help="Command's name",
        )
        parser.add_argument(
            "subcommand_name",
            nargs="?",
            metavar="SUBCOMMAND_NAME",
            help="Subommand's name",
        )

    def execute(self, args: Namespace) -> CommandResult:
        command = args.command_name
        subcommand = args.subcommand_name
        commands = self._assistant.command_executor.commands
        if command:
            if command not in commands:
                return InvalidArgument(
                    error_message=f"Unknown command '{command}'"
                )
            if subcommand:
                parser = commands[command].subparsers.get(subcommand, None)
                if parser is None:
                    return InvalidArgument(
                        error_message=f"Command '{command}' does not have '{subcommand}' subcommand"
                    )
                text = parser.format_help()
            else:
                text = commands[command].parser.format_help()
                if commands[command].subparsers:
                    text += f"\n{command} has the following subcommands:\n\n"
                    for name, subparser in commands[command].subparsers.items():
                        text += f"- {name}: {subparser.description}\n"
                    text += f'\nUse "/help {command} SUBCOMMAND" to get more details.'
            return CommandOutput(content=text)
        command_info = "\n".join(
            sorted(f"- /{c.name}: {c.description}" for c in commands.values())
        )
        return CommandOutput(
            content=_HELP_TEXT.strip().format(commands=command_info)
        )
