import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError

from aideforge.terminal.rich import RichTerminal, _to_rich_theme, _to_rich_style
from aideforge.structs.theme import Theme, TextStyle, DecorationType, TextStyles, theme_collection
from aideforge.abc.terminal import Terminal


class TestRichTerminal:
    def test_creation(self):
        terminal = RichTerminal()
        assert terminal is not None
        assert terminal._use_markup is False

    def test_creation_with_theme(self):
        theme = Theme(name="test")
        terminal = RichTerminal(theme=theme)
        assert terminal is not None

    def test_creation_with_history(self):
        history = ["cmd1", "cmd2", "cmd3"]
        terminal = RichTerminal(history=history)
        # History is loaded into readline

    def test_set_history(self):
        terminal = RichTerminal()
        terminal.set_history(["cmd1", "cmd2"])
        # Should not raise

    def test_set_theme(self):
        terminal = RichTerminal()
        theme = Theme(name="test", text_styles=TextStyles(prompt_string=TextStyle(color="red")))
        terminal.set_theme(theme)
        # Should not raise

    def test_set_theme_monochrome(self):
        terminal = RichTerminal()
        theme = theme_collection["monochrome"]
        terminal.set_theme(theme)
        # Should not raise

    def test_input_without_style(self):
        terminal = RichTerminal()
        with patch.object(terminal, 'input', return_value='test') as mock_input:
            result = terminal.input("prompt> ")
        assert result == "test"

    def test_input_with_style(self):
        terminal = RichTerminal()
        with patch.object(terminal, 'input', return_value='test') as mock_input:
            result = terminal.input("prompt> ", style="prompt_string")
        assert result == "test"

    def test_print(self):
        terminal = RichTerminal()
        terminal.print("hello")
        # Should not raise

    def test_print_with_smart_end(self):
        terminal = RichTerminal()
        terminal.print("hello", smart_end="\n")
        # Should not raise

    def test_print_without_smart_end(self):
        terminal = RichTerminal()
        terminal.print("hello\n", smart_end="\n")
        # Should not raise

    def test_width(self):
        terminal = RichTerminal()
        assert terminal.width > 0


class TestToRichStyle:
    def test_basic_conversion(self):
        style = TextStyle(color="red", bold=True, italic=False)
        rich_style = _to_rich_style(style)
        assert rich_style.color is not None
        assert rich_style.bold is True
        assert rich_style.italic is False

    def test_ansi_color(self):
        style = TextStyle(color=1)
        rich_style = _to_rich_style(style)
        assert rich_style.color is not None

    def test_bgcolor(self):
        style = TextStyle(bgcolor="blue")
        rich_style = _to_rich_style(style)
        assert rich_style.bgcolor is not None

    def test_decoration_underline(self):
        style = TextStyle(decoration=DecorationType.UNDERLINE)
        rich_style = _to_rich_style(style)
        assert rich_style.underline is True

    def test_decoration_overline(self):
        style = TextStyle(decoration=DecorationType.OVERLINE)
        rich_style = _to_rich_style(style)
        assert rich_style.overline is True

    def test_decoration_line_through(self):
        style = TextStyle(decoration=DecorationType.LINE_THROUGH)
        rich_style = _to_rich_style(style)
        assert rich_style.strike is True

    def test_decoration_double_line(self):
        style = TextStyle(decoration=DecorationType.DOUBLE_LINE)
        rich_style = _to_rich_style(style)
        assert rich_style.underline2 is True

    def test_custom_rich_style(self):
        style = TextStyle(custom={"rich": {"blink": True}})
        rich_style = _to_rich_style(style)
        assert rich_style.blink is True

    def test_none_style(self):
        style = TextStyle()
        rich_style = _to_rich_style(style)
        assert rich_style is not None


class TestToRichTheme:
    def test_convert_theme(self):
        theme = Theme(name="test")
        rich_theme = _to_rich_theme(theme)
        assert rich_theme is not None

    def test_convert_theme_with_styles(self):
        theme = Theme(
            name="test",
            text_styles=TextStyles(
                prompt_string=TextStyle(color="green"),
                response=TextStyle(color="yellow", bold=True)
            )
        )
        rich_theme = _to_rich_theme(theme)
        assert rich_theme is not None
        assert "prompt_string" in rich_theme.styles
        assert "response" in rich_theme.styles

    def test_monochrome_theme(self):
        theme = theme_collection["monochrome"]
        rich_theme = _to_rich_theme(theme)
        assert rich_theme is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])