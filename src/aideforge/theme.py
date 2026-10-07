from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ColorTheme",
    "Theme",
    "DefaultColorTheme",
    "DefaultTheme",
    "to_term_color",
    "to_term_colors",
    "to_web_color",
    "to_web_colors",
]

ColorValue = int | str | None
ColorType = ColorValue | tuple[ColorValue, ColorValue]
WebColorValue = str | None
WebColorType = WebColorValue | tuple[WebColorValue, WebColorValue]


class ColorTheme(BaseModel):
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


class Theme(BaseModel):
    """Top-level theme configuration for the CLI.

    Attributes
    ----------
    prompt_string : str or None, optional
        A template for the prompt string. May contain placeholders such as
        ``{agent}`` and ``{model}``.
    colors : ColorTheme, optional
        The color theme to apply. Defaults to a :class:`ColorTheme` instance
        with all colors set to ``None``.
    """

    prompt_string: str | None = None
    colors: ColorTheme = Field(default_factory=ColorTheme)


class MonochromeTheme(ColorTheme):
    """A monochrome (color-less) color theme.

    Attributes
    ----------
    prompt_string : str or None, optional
        A template for the prompt string, showing the agent name.
    """

    prompt_string: str | None = "[{agent}]> "


class DefaultColorTheme(ColorTheme):
    """The default 16-color terminal color theme.
    """

    thinking: ColorType = 8
    thinking_em: ColorType = 7
    response: ColorType = 11
    output_error: ColorType = 15
    error: ColorType = 9


class DefaultTheme(Theme):
    """The default complete theme (prompt template + terminal colors).
    """

    prompt_string: str | None = "[{agent}]> "
    colors: ColorTheme = Field(default_factory=DefaultColorTheme)


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


def to_term_colors(colors: ColorTheme) -> ColorTheme:
    """Convert all colors in a :class:`ColorTheme` to terminal escape sequences.

    Parameters
    ----------
    colors : ColorTheme
        The color theme whose values should be converted. Each color value
        is replaced with its terminal escape sequence (or an empty string
        if the value was ``None``).

    Returns
    -------
    ColorTheme
        A new :class:`ColorTheme` instance (a copy of *colors*) with all
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


def to_web_colors(colors: ColorTheme) -> ColorTheme:
    """Convert all colors in a :class:`ColorTheme` to web-safe hex strings.

    Parameters
    ----------
    colors : ColorTheme
        The color theme whose values should be converted. Each color value
        is replaced with its web-safe hex equivalent (or an empty string
        if the value was ``None``).

    Returns
    -------
    ColorTheme
        A new :class:`ColorTheme` instance (a copy of *colors*) with all
        values replaced by web-safe hex color strings.
    """
    colors = colors.model_copy()
    for name in colors.model_fields:
        val = getattr(colors, name)
        setattr(colors, name, to_web_color(val) or "")
    return colors
