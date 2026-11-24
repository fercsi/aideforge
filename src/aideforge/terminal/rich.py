import readline
import sys
from typing import Any, Literal, TypeVar

from rich.style import Style as RichStyle
from rich.theme import Theme as RichTheme, ThemeStackError
from rich.color import Color as RichColor
from rich.console import Console as RichConsole

from aideforge.abc.terminal import Terminal
from aideforge.structs.theme import TextStyle, Theme

__all__ = ["RichTerminal"]


class RichTerminal(RichConsole, Terminal):
    def __init__(
        self,
        *,
        color_system: Literal[
            "auto", "standard", "256", "truecolor", "windows"
        ] = "auto",
        theme: Theme | None = None,
        history: list[str] | None = None,
    ) -> None:
        # Rich theme is stacked. We use a single theme always, replacable
        # So set empty theme to the bottom
        RichConsole.__init__(
            self,
            color_system=color_system,
            theme=RichTheme({}),
        )
        if theme is not None:
            self.set_theme(theme)
        if history is not None:
            self.set_history(history)

    def set_history(self, history: list[str]) -> None:
        readline.clear_history()
        for item in history:
            readline.add_history(item)
        readline.set_auto_history(True)

    def set_theme(self, theme: Theme) -> None:
        rich_theme = _to_rich_theme(theme)
        try:
            self.pop_theme()
        except ThemeStackError:  # on first call no theme, no problem
            pass
        self.push_theme(rich_theme)

    def input(self, prompt: str, *, style: str | None = None, **kwargs) -> str:
        if style is not None:
            prompt = f"[{style}]{prompt}[/{style}]"
        return RichConsole.input(self, prompt)

    def print(self, *args, smart_end: str | None = None, **kwargs) -> None:
        if smart_end:
            if not isinstance(args[-1], str) or not args[-1].endswith(
                smart_end
            ):
                kwargs["end"] = smart_end
            else:
                kwargs["end"] = ""
        RichConsole.print(self, *args, **kwargs)


def _to_rich_theme(theme: Theme) -> RichTheme:
    styles = theme.text_styles
    rich_styles = {}
    for name in type(styles).model_fields:
        style = getattr(styles, name)
        if style is not None:
            rich_styles[name] = _to_rich_style(style)
        else:
            rich_styles[name] = RichStyle()
    return RichTheme(styles=rich_styles)


def _to_rich_style(style: TextStyle) -> RichStyle:
    rich_style = {}
    if style.color is not None:
        if isinstance(style.color, int):
            rich_style["color"] = RichColor.from_ansi(style.color)
        else:
            assert isinstance(style.color, str)
            rich_style["color"] = RichColor.parse(style.color)
    if style.bgcolor is not None:
        if isinstance(style.bgcolor, int):
            rich_style["bgcolor"] = RichColor.from_ansi(style.bgcolor)
        else:
            assert isinstance(style.bgcolor, str)
            rich_style["bgcolor"] = RichColor.parse(style.bgcolor)
    if style.italic is not None:
        rich_style["italic"] = style.italic
    if style.bold is not None:
        rich_style["bold"] = style.bold
    if style.decoration is not None:
        rich_style["underline"] = False
        rich_style["underline2"] = False
        rich_style["strike"] = False
        rich_style["overline"] = False
        if style["decoration"] == DecorationType.UNDERLINE:
            rich_style["underline"] = True
        elif style.decoration == DecorationType.UNDERLINE2:
            rich_style["underline2"] = True
        elif style.decoration == DecorationType.LINE_THROUGH:
            rich_style["strike"] = True
        elif style.decoration == DecorationType.OVERLINE:
            rich_style["overline"] = True
    if style.custom is not None and "rich" in style.custom:
        rich_style.update(style.custom["rich"])
    return RichStyle(**rich_style)
