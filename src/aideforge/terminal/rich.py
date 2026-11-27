import readline
import sys
from typing import Any, Literal, TypeVar


from rich import errors, markup
from rich.color import Color as RichColor
from rich.console import Console as RichConsole
from rich.style import Style as RichStyle
from rich.theme import Theme as RichTheme, ThemeStackError
from aideforge.abc.terminal import Terminal
from aideforge.structs.theme import DecorationType, TextStyle, Theme

__all__ = ["RichTerminal"]


class RichTerminal(RichConsole, Terminal):
    """Rich-based terminal implementation.

    Extends Rich Console with history management and theme support.
    """

    def __init__(
        self,
        *,
        color_system: Literal[
            "auto", "standard", "256", "truecolor", "windows"
        ] = "auto",
        theme: Theme | None = None,
        history: list[str] | None = None,
        use_markup: bool = False,
    ) -> None:
        """Initialize the Rich terminal.

        Parameters
        ----------
        color_system : Literal["auto", "standard", "256", "truecolor", "windows"], optional
            Color system to use, by default "auto".
        theme : Theme or None, optional
            Theme to apply, by default None.
        history : list[str] or None, optional
            Command history to load, by default None.
        use_markup : bool, optional
            Whether to use Rich markup for rendering, by default False.
        """
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
        self._use_markup = use_markup

    def set_history(self, history: list[str]) -> None:
        """Set the command history.

        Parameters
        ----------
        history : list[str]
            List of history entries to load.
        """
        readline.clear_history()
        for item in history:
            readline.add_history(item)
        readline.set_auto_history(True)

    def set_theme(self, theme: Theme) -> None:
        """Apply a color theme to the terminal.

        Parameters
        ----------
        theme : Theme
            Theme configuration to apply.
        """
        rich_theme = _to_rich_theme(theme)
        try:
            self.pop_theme()
        except ThemeStackError:  # on first call no theme, no problem
            pass
        self.push_theme(rich_theme)

    def input(self, prompt: str, *, style: str | None = None, **kwargs) -> str:
        """Read a line of input from the user.

        Parameters
        ----------
        prompt : str
            Prompt string to display.
        style : str or None, optional
            Style name to apply to the prompt, by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.

        Returns
        -------
        str
            The user's input line.
        """
        use_markup = kwargs.setdefault("markup", self._use_markup)
        if not use_markup:
            prompt = markup.escape(prompt)
        kwargs["markup"] = True
        if style is not None:
            prompt = f"[{style}]{prompt}[/{style}]"
        if style is not None and isinstance(style, str):
            style_names = style.split()
            for name in style_names:
                try:
                    self.get_style(name)
                except errors.MissingStyle:
                    raise errors.MissingStyle(f"Failed to get style '{name}'") from None
            style_start = "".join(f"[{n}]" for n in style_names)
            style_end = "[/]" * len(style_names)
            prompt = style_start + prompt + style_end
        return RichConsole.input(self, prompt)

    def print(self, *args, smart_end: str | None = None, **kwargs) -> None:
        """Print output to the terminal.

        Parameters
        ----------
        *args : Any
            Objects to print.
        smart_end : str or None, optional
            Ending string if the last argument doesn't already end with it,
            by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.
        """
        if smart_end:
            if not isinstance(args[-1], str) or not args[-1].endswith(
                smart_end
            ):
                kwargs["end"] = smart_end
            else:
                kwargs["end"] = ""
        kwargs.setdefault("markup", self._use_markup)
        kwargs.setdefault("highlight", False)
        style = kwargs.get("style")
        if style is not None and isinstance(style, str):
            style_names = style.split()
            styles = []
            for name in style_names:
                try:
                    styles.append(self.get_style(name))
                except errors.MissingStyle:
                    raise errors.MissingStyle(f"Failed to get style '{name}'") from None
            style = styles[0]
            for s in styles[1:]:
                style = style + s
            kwargs["style"] = style

        RichConsole.print(self, *args, **kwargs)


def _to_rich_theme(theme: Theme) -> RichTheme:
    """Convert a Theme to a Rich Theme.

    Parameters
    ----------
    theme : Theme
        Theme to convert.

    Returns
    -------
    RichTheme
        Rich-compatible theme.
    """
    styles = theme.text_styles
    rich_styles = {  # some basic style components
        "bold": RichStyle(bold=True),
        "italic": RichStyle(italic=True),
        "underline": RichStyle(underline=True),
    }
    for name in type(styles).model_fields:
        style = getattr(styles, name)
        if style is not None:
            rich_styles[name] = _to_rich_style(style)
        else:
            rich_styles[name] = RichStyle()
    return RichTheme(styles=rich_styles)


def _to_rich_style(style: TextStyle) -> RichStyle:
    """Convert a TextStyle to a Rich Style.

    Parameters
    ----------
    style : TextStyle
        TextStyle to convert.

    Returns
    -------
    RichStyle
        Rich-compatible style.
    """
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
        if style.decoration == DecorationType.UNDERLINE:
            rich_style["underline"] = True
        elif style.decoration == DecorationType.DOUBLE_LINE:
            rich_style["underline2"] = True
        elif style.decoration == DecorationType.LINE_THROUGH:
            rich_style["strike"] = True
        elif style.decoration == DecorationType.OVERLINE:
            rich_style["overline"] = True
    if style.custom is not None and "rich" in style.custom:
        rich_style.update(style.custom["rich"])
    return RichStyle(**rich_style)