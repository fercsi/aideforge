__all__ = (
    "BaseCommand",
    "CommandError",
    "CommandExecutor",
    "CommandOutput",
    "CommandResult",
    "CommandSuccess",
    "ExitAssistant",
    "default_commands",
)

import argparse
import importlib
import inspect
import re
from collections.abc import Sequence
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Annotated, Any, Literal, Self, Type

from aideforge.status import (
    UnknownCommand,
    InvalidArgument,
    CommandExecutionError,
    Success,
)


class ExitAssistant(BaseModel):
    status: Literal["exit"] = "exit"


class CommandOutput(BaseModel):
    status: Literal["ouput"] = "output"
    content: str


CommandError = Annotated[
    UnknownCommand | InvalidArgument | CommandExecutionError,
    Field(descriminator="error_code"),
]

CommandSuccess = Annotated[
    Success | CommandOutput | ExitAssistant, Field(descriminator="status")
]

CommandResult = Annotated[
    CommandSuccess | CommandError, Field(descriminator="status")
]


class CommandArgumentError(BaseException):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CommandArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)

    def add_subparsers(self, *args: Any, **kwargs: Any) -> argparse.Action:
        kwargs.setdefault("required", True)
        kwargs.setdefault("metavar", "SUBCOMMAND")
        kwargs.setdefault("help", "%(choices)s")
        return super().add_subparsers(*args, **kwargs)

    def error(self, message):
        raise CommandArgumentError(message)


class BaseCommand:
    name: str
    description: str

    def __init__(self, assistant):
        self._assistant = assistant
        self.parser = CommandArgumentParser(
            prog="/" + self.name,
            description=self.description,
        )
        self.parser.set_defaults(func=self.execute)
        self.subparsers = {}
        self.setup(parser=self.parser, subparsers=self.subparsers)

    def parse_args(self, args: Sequence[str]) -> argparse.Namespace:
        try:
            args = self.parser.parse_args(args)
        except CommandArgumentError as exc:
            return None, exc
        return args, None

    def setup(
        self,
        parser: CommandArgumentParser,
        subparsers: dict[str, CommandArgumentParser],
    ) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> CommandOutput:
        pass


class CommandExecutor:
    def __init__(self, assistant):
        self.commands: dict[str, BaseCommand] = {}
        self._assistant = assistant

    def register_command(self, command_cls: Type[BaseCommand]) -> None:
        """Register a command class using its name attribute."""
        if not hasattr(command_cls, "name") or not command_cls.name:
            raise ValueError(
                "Command class must define a valid 'name' attribute."
            )

        if command_cls.name in self.commands:
            raise ValueError(
                f"Command '{command_cls.name}' is already registered."
            )

        self.commands[command_cls.name] = command_cls(self._assistant)

    def execute_command(self, name: str, arg_list: list[str]) -> CommandResult:
        """Find the registered command and execute it with *args and **kwargs."""
        if name not in self.commands:
            return UnknownCommand(error_message=f"Command '{name}' not found.")

        command = self.commands[name]
        args, err = command.parse_args(arg_list)
        if err:
            return InvalidArgument(error_message=str(err))
        try:
            return args.func(args)
        except Exception as exc:
            return CommandExecutionError(error_message=str(exc))


def _import_commands() -> list[BaseCommand]:
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
