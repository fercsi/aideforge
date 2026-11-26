from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal, TypeAlias, Union

from pydantic import BaseModel, Field

from .abc import BaseStruct

__all__ = [
    "ColorType",
    "ColorValue",
    "Theme",
    "ThemeColor",
    "WebColorType",
    "WebColorValue",
    "theme_collection",
    "to_term_color",
    "to_term_colors",
    "to_web_color",
    "to_web_colors",
]

ColorValue : TypeAlias = Union[int, str, None]
"""Color value type.

The color value can be an integer representing a color from the 16-color
palette or the 256-color terminal palette, or a web-style hexadecimal color
string (e.g. ``"#xxxxxx"``).

If the value is None, the default color is used.
"""

ColorType : TypeAlias = Union[ColorValue, tuple[ColorValue, ColorValue]]
"""Color type.

Unlike ``ColorValue``, this value can specify both a foreground and a
background color using a two-element tuple ``(fg, bg)``. Either element
can be ``None``, indicating that the default color should be used for that
component.

If the value is not a ``tuple``, it must be a valid ``ColorValue`` and
specifies only the foreground color. A value of ``value`` is equivalent to
``(value, None)``.
"""

WebColorValue: TypeAlias = Union[str, None]
"""Web color value type.

Always a web-style hexadecimal color string (e.g. ``"#xxxxxx"``).
"""

WebColorType: TypeAlias = Union[WebColorValue, tuple[WebColorValue, WebColorValue]]
"""Web color type.

This value can specify both a foreground and a background color using a
two-element tuple ``(fg, bg)``. Either element can be ``None``, indicating
that the default color should be used for that component.

If the value is not a ``tuple``, it must be a valid ``WebColorValue`` and
specifies only the foreground color. A value of ``value`` is equivalent to
``(value, None)``.
"""

class DecorationType(StrEnum):
    UNDERLINE = "underline"
    OVERLINE = "overline"
    LINE_THROUGH = "line-through"
    DOUBLE_LINE = "double-line"

class TextStyle(BaseModel):
    """
    Colors: Each attribute accepts a number (0..255 for 16 or 256-color palette),
    an ANSI color string (``"#rrggbb"`` format), or ``None`` for no color.
    A tuple pairs a foreground color with a background color, single value
    refers to a forground color (like ``(color, None)``).
    """

    color: int | str | None = None
    bgcolor: int | str | None = None
    bold: bool | None = None
    italic: bool | None = None
    decoration: DecorationType | None = None
    custom: dict[str, dict[str, Any]] | None = None
 
class TextStyles(BaseModel):
    """Text style configuration for different parts of the terminal interface.

    Attributes
    ----------
    prompt_string : TextStyle or None, optional
        Style for the prompt string.
    tool_prompt_string : TextStyle or None, optional
        Style for tool permission prompts.
    input : TextStyle or None, optional
        Style for user input text/prompt.
    tool_input : TextStyle or None, optional
        Style for tool input text.
    welcome_message : TextStyle or None, optional
        Style for the welcome message.
    llm_thinking : TextStyle or None, optional
        Style for the assistant's thinking output.
    llm_id : TextStyle or None, optional
        Style for LLM message IDs.
    llm_tool_call : TextStyle or None, optional
        Style for LLM tool calls.
    llm_tool_output : TextStyle or None, optional
        Style for LLM tool output.
    response : TextStyle or None, optional
        Style for the assistant's response text.
    output : TextStyle or None, optional
        Style for general output (e.g. command) text.
    output_error : TextStyle or None, optional
        Style for error output text.
    warning : TextStyle or None, optional
        Style for application level warning messages.
    error : TextStyle or None, optional
        Style for application level error messages.
    """
    # Default values must be always None!

    # Input
    prompt_string: TextStyle | None = None
    tool_prompt_string: TextStyle | None = None
    input: TextStyle | None = None
    tool_input: TextStyle | None = None
    # Assistant
    welcome_message: TextStyle | None = None
    llm_thinking: TextStyle | None = None
    llm_id: TextStyle | None = None
    llm_tool_call: TextStyle | None = None
    llm_tool_output: TextStyle | None = None
    response: TextStyle | None = None
    # Output
    output: TextStyle | None = None
    output_error: TextStyle | None = None
    # Application
    warning: TextStyle | None = None
    error: TextStyle | None = None


class Theme(BaseStruct):
    """Theme configuration for the CLI and other implementations.

    Inherited fixed attributes and metadata:

    - **kind**: ``"theme"`` (kinds: ``"themes"``)
    - **version**: ``1``
    - **scope**: ``"global"``
    - **storage_name**: ``struct.name``

    Attributes
    ----------
    name : str
        Name of the theme to identify with
    colors : ThemeColor
        The color theme to apply. Defaults to a :class:`ThemeColor` instance
        with all colors set to ``None``.
    """

    kind: Literal["theme"] = "theme"
    version: Literal[1] = 1
    name: str = Field(..., description="Name of the theme to identify with")
    text_styles: TextStyles = Field(
        default_factory = TextStyles,
        description="",
    )
    # TODO: fonts, borders, styles... for web maybe Annotated of WebTheme/CliTheme

    # Metadata
    scope: ClassVar[Literal["global"]] = "global"
    storage_name: ClassVar[str] = "{struct.name}"
    kinds: ClassVar[Literal["themes"]] = "themes"


theme_collection: dict[str, Theme] = {
    "default": Theme(
        name="default",
        text_styles={
            "welcome_message": {"color":"bright_yellow"},
            "tool_prompt_string": {"color":"bright_cyan"},
            "llm_id": {"color":"white"},
            "llm_thinking": {"color":"bright_black"},
            "llm_tool_call": {"color":"cyan"},
            "llm_tool_output": {"color":"cyan"},
            "response": {"color":"bright_yellow"},
            "output_error": {"color":"bright_white"},
            "error": {"color":"bright_red"},
        },
    ),
    "monochrome": Theme(
        name="monochrome",
        text_styles={},
    ),
}


def to_term_color(color: ColorType, *, background: bool = False) -> str | None:
    """Convert a color value to a terminal escape sequence.

    Parameters
    ----------
    color : ColorType
        The color value to convert. May be an ANSI index (0..255),
        a hex string (``"#rrggbb"``), a tuple of ``(foreground, background)``,
        or ``None``.
    background : bool, optional
        Whether to emit the color as a background color, by default ``False``.

    Returns
    -------
    str or None
        The ANSI escape sequence as a string, or ``None`` if *color* is
        ``None``. For tuple inputs, the combined foreground/background
        sequence is returned.

    Raises
    ------
    ValueError
        If *color* is not a valid color value.
    """
    if color is None:
        return None
    if isinstance(color, (tuple, list)):
        fg = to_term_color(color[0])
        bg = to_term_color(color[1], background=True)
        if fg is None:
            return bg
        if bg is None:
            return fg
        return f"{fg[:-1]};{bg[2:]}"
    sel = "4" if background else "3"
    try:
        if isinstance(color, int):
            if color < 0:
                raise ValueError("")
            if color < 8:
                return f"\x1b[{sel}{color}m"
            if not background and color < 16:
                return f"\x1b[1;3{color-8}m"
            if color < 256:
                return f"\x1b[{sel}8;5;{color}m"
        if isinstance(color, str) and len(color) == 7 and color[0] == "#":
            rgb = ";".join(str(n) for n in bytes.fromhex(color[1:]))
            return f"\x1b[{sel}8;2;{rgb}m"
        raise ValueError("")
    except:
        raise ValueError(f"Invalid color value '{color}'")


def to_term_colors(colors: ThemeColor) -> ThemeColor:
    """Convert all colors in a :class:`ThemeColor` to terminal escape sequences.

    Parameters
    ----------
    colors : ThemeColor
        The color theme whose values should be converted. Each color value
        is replaced with its terminal escape sequence (or an empty string
        if the value was ``None``).

    Returns
    -------
    ThemeColor
        A new :class:`ThemeColor` instance (a copy of *colors*) with all
        values replaced by terminal escape sequences.
    """
    colors = colors.model_copy()
    for name in colors.model_fields:
        val = getattr(colors, name)
        setattr(colors, name, to_term_color(val) or "")
    return colors


def to_web_color(color: ColorType) -> WebColorType:
    """Convert a color value to a web-safe hex color string.

    Parameters
    ----------
    color : ColorType
        The color value to convert. May be an ANSI index (0..255), a hex
        string (``"#rrggbb"``), or a tuple of ``(foreground, background)``.

    Returns
    -------
    WebColorType
        A hex color string (e.g. ``"#rrggbb"``), ``None`` if *color* is
        ``None``, or a tuple of ``(fg, bg)`` for tuple inputs.

    Raises
    ------
    ValueError
        If *color* is not a valid color value or is out of range.
    """
    if color is None:
        return None
    if isinstance(color, (tuple, list)) and len(color) == 2:
        fg = to_web_color(color[0])
        bg = to_web_color(color[1])
        return (fg, bg)

    try:
        if isinstance(color, int):
            if not 0 <= color <= 255:
                raise ValueError("index must be between 0 and 255")

            # ANSI base colors
            if color < 16:
                r = 1 if color & 1 else 0
                g = 1 if color & 2 else 0
                b = 1 if color & 4 else 0

                value = 255 if color >= 8 else 128

                return f"#{r * value:02x}{g * value:02x}{b * value:02x}"

            # 6x6x6 color cube
            if color < 232:
                color -= 16
                r, color = divmod(color, 36)
                g, b = divmod(color, 6)

                values = (0, 95, 135, 175, 215, 255)
                return f"#{values[r]:02x}{values[g]:02x}{values[b]:02x}"

            # Grayscale
            value = 8 + (color - 232) * 10
            return f"#{value:02x}{value:02x}{value:02x}"

        if isinstance(color, str) and len(color) == 7 and color[0] == "#":
            return color
    except:
        pass
    raise ValueError(f"Invalid color value '{color}'")


def to_web_colors(colors: ThemeColor) -> ThemeColor:
    """Convert all colors in a :class:`ThemeColor` to web-safe hex strings.

    Parameters
    ----------
    colors : ThemeColor
        The color theme whose values should be converted. Each color value
        is replaced with its web-safe hex equivalent (or an empty string
        if the value was ``None``).

    Returns
    -------
    ThemeColor
        A new :class:`ThemeColor` instance (a copy of *colors*) with all
        values replaced by web-safe hex color strings.
    """
    colors = colors.model_copy()
    for name in colors.model_fields:
        val = getattr(colors, name)
        setattr(colors, name, to_web_color(val) or "")
    return colors
