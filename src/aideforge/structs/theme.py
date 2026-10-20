from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .abc import BaseStruct

__all__ = [
    "ThemeColor",
    "Theme",
    "theme_collection",
    "to_term_color",
    "to_term_colors",
    "to_web_color",
    "to_web_colors",
]

ColorValue = int | str | None
ColorType = ColorValue | tuple[ColorValue, ColorValue]
WebColorValue = str | None
WebColorType = WebColorValue | tuple[WebColorValue, WebColorValue]


class ThemeColor(BaseModel):
    """Color configuration for different parts of the terminal interface.

    Each attribute accepts a number (0..255 for 16 or 256-color palette),
    an ANSI color string (``"#rrggbb"`` format), or ``None`` for no color.
    A tuple pairs a foreground color with a background color, single value
    refers to a forground color (like ``(color, None)``).

    Attributes
    ----------
    prompt_string : ColorType, optional
        Color for the prompt string.
    input : ColorType, optional
        Color for user input text/prompt.
    thinking : ColorType, optional
        Color for the assistant's thinking output.
    thinking_em : ColorType, optional
        Color for emphasized thinking text.
    response : ColorType, optional
        Color for the assistant's response text.
    output : ColorType, optional
        Color for general output (e.g. command) text.
    output_error : ColorType, optional
        Color for error output text.
    warning : ColorType, optional
        Color for application level warning messages.
    error : ColorType, optional
        Color for application level error messages.
    """

    # Input
    prompt_string: ColorType = None
    input: ColorType = None
    # Assistant
    thinking: ColorType = None
    thinking_em: ColorType = None
    response: ColorType = None
    # Output
    output: ColorType = None
    output_error: ColorType = None
    # Application
    warning: ColorType = None
    error: ColorType = None


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
    colors: ThemeColor = Field(default_factory=ThemeColor)
    # TODO: fonts, borders, styles... for web maybe Annotated of WebTheme/CliTheme

    # Metadata
    scope: ClassVar[Literal["global"]] = "global"
    storage_name: ClassVar[str] = "{struct.name}"
    kinds: ClassVar[Literal["themes"]] = "themes"


theme_collection: dict[str, Theme] = {
    "default": Theme(
        name="default",
        colors={
            "thinking": 8,
            "thinking_em": 7,
            "response": 11,
            "output_error": 15,
            "error": 9,
        },
    ),
    "monochrome": Theme(
        name="monochrome",
        colors={},
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
