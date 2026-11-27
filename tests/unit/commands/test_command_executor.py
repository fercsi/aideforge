import pytest
from unittest.mock import MagicMock

from aideforge.command_executor import CommandExecutor
from aideforge.abc.command import BaseCommand
from aideforge.abc.assistant import AbstractAssistant
from aideforge.results import (
    Success, CommandResult, CommandExecutionError, 
    InternalError, InvalidArgument, UnknownCommand
)
from aideforge.abc.command import CommandArgumentError


class TestCommandExecutor:
    def setup_method(self):
        self.mock_assistant = MagicMock(spec=AbstractAssistant)

    def test_creation(self):
        executor = CommandExecutor(self.mock_assistant)
        assert executor._assistant == self.mock_assistant
        assert executor._commands == {}
        assert executor._aliases == {}

    def test_creation_invalid_assistant_raises(self):
        with pytest.raises(ValueError, match="Expected an AbstractAssistant"):
            CommandExecutor("not an assistant")

    def test_register_command(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        assert "test" in executor._commands
        assert executor._commands["test"].name == "test"

    def test_register_command_no_name_raises(self):
        class TestCommand(BaseCommand):
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        with pytest.raises(ValueError, match="Command must define a valid 'name' attribute"):
            executor.register_command(TestCommand)

    def test_register_command_empty_name_raises(self):
        class TestCommand(BaseCommand):
            name = ""
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        with pytest.raises(ValueError, match="Command must define a valid 'name' attribute"):
            executor.register_command(TestCommand)

    def test_register_command_duplicate_raises(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        with pytest.raises(ValueError, match="Command 'test' is already registered"):
            executor.register_command(TestCommand)

    def test_register_command_with_aliases(self):
        class TestCommand(BaseCommand):
            name = "test"
            aliases = ["t", "test_alias"]
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        assert "test" in executor._commands
        assert "t" in executor._aliases
        assert "test_alias" in executor._aliases
        assert executor._aliases["t"] is executor._commands["test"]

    def test_register_command_invalid_aliases_raises(self):
        class TestCommand(BaseCommand):
            name = "test"
            aliases = "not a list"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        with pytest.raises(ValueError, match="Expected list of aliases"):
            executor.register_command(TestCommand)

    def test_execute_command_success(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                parser.add_argument("--value", type=str, default="default")
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        result = executor.execute_command("test", ["--value", "hello"])
        assert isinstance(result, Success)

    def test_execute_command_unknown(self):
        executor = CommandExecutor(self.mock_assistant)
        
        result = executor.execute_command("unknown", [])
        assert isinstance(result, UnknownCommand)
        assert "not found" in result.error_message

    def test_execute_command_invalid_args(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                parser.add_argument("--required", required=True)
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        result = executor.execute_command("test", [])
        assert isinstance(result, InvalidArgument)
        assert "required" in result.error_message.lower()

    def test_execute_command_execution_error(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                raise ValueError("Execution failed")
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        result = executor.execute_command("test", [])
        assert isinstance(result, CommandExecutionError)
        assert "Execution failed" in result.error_message

    def test_execute_command_with_exception(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                parser.add_argument("--fail", action="store_true")
            
            def execute(self, args):
                if args.fail:
                    raise RuntimeError("Test error")
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        result = executor.execute_command("test", ["--fail"])
        assert isinstance(result, CommandExecutionError)
        assert "Test error" in result.error_message

    def test_commands_property(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        commands = executor.commands
        assert "test" in commands
        assert commands["test"].name == "test"


class TestCommandExecutorWithSubcommands:
    def setup_method(self):
        self.mock_assistant = MagicMock(spec=AbstractAssistant)

    def test_execute_subcommand(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                subparsers_action = parser.add_subparsers(required=False)
                sub = subparsers_action.add_parser("sub1", description="Sub 1")
                sub.set_defaults(func=self.sub1)
                subparsers["sub1"] = sub
            
            def sub1(self, args):
                from aideforge.results import CommandOutput
                return CommandOutput(content="sub1 executed")
            
            def execute(self, args):
                return Success()
        
        executor = CommandExecutor(self.mock_assistant)
        executor.register_command(TestCommand)
        
        result = executor.execute_command("test", ["sub1"])
        assert hasattr(result, 'content')
        assert result.content == "sub1 executed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])