import pytest
from argparse import Namespace

from aideforge.abc.command import CommandArgumentParser, BaseCommand, CommandArgumentError
from aideforge.results import Success, CommandResult
from aideforge.abc.assistant import AbstractAssistant
from unittest.mock import MagicMock


class TestCommandArgumentParser:
    def test_creation(self):
        parser = CommandArgumentParser(prog="test", description="Test parser")
        assert parser.prog == "test"
        assert parser.description == "Test parser"
        # With add_help=False, there's no -h/--help action
        help_actions = [a for a in parser._actions if a.dest == 'help']
        assert len(help_actions) == 0

    def test_add_subparsers_defaults(self):
        parser = CommandArgumentParser()
        subparsers = parser.add_subparsers()
        # Check defaults were set
        assert subparsers.required is True
        assert subparsers.metavar == "SUBCOMMAND"
        assert subparsers.help == "%(choices)s"

    def test_error_raises_exception(self):
        parser = CommandArgumentParser()
        with pytest.raises(CommandArgumentError, match="test error"):
            parser.error("test error")

    def test_parse_args_success(self):
        parser = CommandArgumentParser()
        parser.add_argument("--test", type=str)
        result = parser.parse_args(["--test", "value"])
        assert result.test == "value"

    def test_parse_args_error(self):
        parser = CommandArgumentParser()
        parser.add_argument("--test", required=True)
        with pytest.raises(CommandArgumentError):
            parser.parse_args([])


class TestBaseCommand:
    def setup_method(self):
        self.mock_assistant = MagicMock(spec=AbstractAssistant)

    def test_creation(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        assert cmd.name == "test"
        assert cmd.description == "Test command"
        assert cmd._assistant == self.mock_assistant
        assert cmd.parser is not None

    def test_creation_with_aliases(self):
        class TestCommand(BaseCommand):
            name = "test"
            aliases = ["t", "test_alias"]
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        assert cmd.aliases == ["t", "test_alias"]

    def test_creation_with_category(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            category = "custom"
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        assert cmd.category == "custom"

    def test_setup_called_during_init(self):
        setup_called = []
        
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                setup_called.append(True)
                parser.add_argument("--custom", type=str)
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        assert len(setup_called) == 1

    def test_parse_args_success(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                parser.add_argument("--value", type=str)
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        args, error = cmd.parse_args(["--value", "hello"])
        assert error is None
        assert args.value == "hello"

    def test_parse_args_error(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                parser.add_argument("--required", required=True)
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        args, error = cmd.parse_args([])
        assert args is None
        assert isinstance(error, CommandArgumentError)

    def test_execute_default(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        result = cmd.execute(Namespace())
        assert isinstance(result, Success)

    def test_subparsers(self):
        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            
            def setup(self, parser, subparsers, **kwargs):
                subparsers_action = parser.add_subparsers(required=False)
                subparser = subparsers_action.add_parser("sub1", description="Sub 1")
                subparser.set_defaults(func=self.sub1)
                subparsers["sub1"] = subparser
            
            def sub1(self, args):
                return Success()
            
            def execute(self, args):
                return Success()
        
        cmd = TestCommand(self.mock_assistant)
        assert "sub1" in cmd.subparsers


class TestCommandArgumentError:
    def test_creation(self):
        exc = CommandArgumentError("test message")
        assert str(exc) == "test message"
        assert exc.message == "test message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])