import pytest
from pydantic import ValidationError

from aideforge.structs.theme import (
    DecorationType,
    TextStyle,
    TextStyles,
    Theme,
    theme_collection,
)


class TestDecorationType:
    def test_values(self):
        assert DecorationType.UNDERLINE == "underline"
        assert DecorationType.OVERLINE == "overline"
        assert DecorationType.LINE_THROUGH == "line-through"
        assert DecorationType.DOUBLE_LINE == "double-line"


class TestTextStyle:
    def test_default_values(self):
        ts = TextStyle()
        assert ts.color is None
        assert ts.bgcolor is None
        assert ts.bold is None
        assert ts.italic is None
        assert ts.decoration is None
        assert ts.custom is None

    def test_with_int_color(self):
        ts = TextStyle(color=1, bgcolor=2)
        assert ts.color == 1
        assert ts.bgcolor == 2

    def test_with_hex_color(self):
        ts = TextStyle(color="#ff0000", bgcolor="#00ff00")
        assert ts.color == "#ff0000"
        assert ts.bgcolor == "#00ff00"

    def test_with_bold_italic(self):
        ts = TextStyle(bold=True, italic=False)
        assert ts.bold is True
        assert ts.italic is False

    def test_with_decoration(self):
        ts = TextStyle(decoration=DecorationType.UNDERLINE)
        assert ts.decoration == DecorationType.UNDERLINE

    def test_with_custom(self):
        ts = TextStyle(custom={"rich": {"style": "bold"}})
        assert ts.custom == {"rich": {"style": "bold"}}


class TestTextStyles:
    def test_default_values(self):
        tss = TextStyles()
        assert tss.prompt_string is None
        assert tss.tool_prompt_string is None
        assert tss.input is None
        assert tss.tool_input is None
        assert tss.welcome_message is None
        assert tss.llm_thinking is None
        assert tss.llm_id is None
        assert tss.llm_tool_call is None
        assert tss.llm_tool_output is None
        assert tss.response is None
        assert tss.output is None
        assert tss.output_error is None
        assert tss.warning is None
        assert tss.error is None

    def test_with_dict_values(self):
        tss = TextStyles(
            prompt_string={"color": "green"},
            response={"color": "yellow", "bold": True}
        )
        assert tss.prompt_string.color == "green"
        assert tss.response.color == "yellow"
        assert tss.response.bold is True

    def test_with_textstyle_objects(self):
        tss = TextStyles(
            prompt_string=TextStyle(color="red"),
            welcome_message=TextStyle(color="blue", bold=True)
        )
        assert isinstance(tss.prompt_string, TextStyle)
        assert tss.prompt_string.color == "red"
        assert tss.welcome_message.color == "blue"
        assert tss.welcome_message.bold is True


class TestTheme:
    def test_creation(self):
        theme = Theme(name="test_theme")
        assert theme.kind == "theme"
        assert theme.version == 1
        assert theme.name == "test_theme"
        assert isinstance(theme.text_styles, TextStyles)

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            Theme()

    def test_metadata(self):
        assert Theme.scope == "global"
        assert Theme.storage_name == "{struct.name}"
        assert Theme.kinds == "themes"

    def test_theme_collection(self):
        assert "default" in theme_collection
        assert "monochrome" in theme_collection
        assert theme_collection["default"].name == "default"
        assert theme_collection["monochrome"].name == "monochrome"

    def test_default_theme_styles(self):
        default = theme_collection["default"]
        assert default.text_styles.welcome_message is not None
        assert default.text_styles.welcome_message.color == "bright_yellow"
        assert default.text_styles.tool_prompt_string.color == "bright_cyan"
        assert default.text_styles.response.color == "bright_yellow"
        assert default.text_styles.error.color == "bright_red"
        assert default.text_styles.llm_id.color == "white"
        assert default.text_styles.llm_thinking.color == "bright_black"
        assert default.text_styles.llm_tool_call.color == "cyan"
        assert default.text_styles.llm_tool_output.color == "cyan"
        assert default.text_styles.output_error.color == "bright_white"

    def test_monochrome_theme_empty(self):
        mono = theme_collection["monochrome"]
        assert mono.text_styles.prompt_string is None
        assert mono.text_styles.welcome_message is None

    def test_with_custom_text_styles(self):
        theme = Theme(
            name="custom",
            text_styles=TextStyles(
                prompt_string=TextStyle(color="magenta"),
                error=TextStyle(color="red", bold=True)
            )
        )
        assert theme.text_styles.prompt_string.color == "magenta"
        assert theme.text_styles.error.color == "red"
        assert theme.text_styles.error.bold is True