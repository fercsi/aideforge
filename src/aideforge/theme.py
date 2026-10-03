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
    # number: 0..255
    # str: "#xxxxxx"
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
    prompt_string: str | None = None
    colors: ColorTheme = Field(default_factory=ColorTheme)


class MonochromeTheme(ColorTheme):
    prompt_string: str | None = "[{agent}]> "


class DefaultColorTheme(ColorTheme):
    thinking: ColorType = 8
    thinking_em: ColorType = 7
    response: ColorType = 11
    output_error: ColorType = 15
    error: ColorType = 9


class DefaultTheme(Theme):
    prompt_string: str | None = "[{agent}]> "
    colors: ColorTheme = Field(default_factory=DefaultColorTheme)


def to_term_color(color: ColorType, *, background: bool = False) -> str | None:
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
    colors = colors.model_copy()
    for name in colors.model_fields:
        val = getattr(colors, name)
        setattr(colors, name, to_term_color(val) or "")
    return colors


def to_web_color(color: ColorType, *, background: bool = False) -> WebColorType:
    if color is None:
        return None
    if isinstance(color, (tuple, list)):
        fg = to_web_color(color[0])
        bg = to_web_color(color[1], background=True)
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
        raise ValueError("")
    except:
        raise ValueError(f"Invalid color value '{color}'")


def to_web_colors(colors: ColorTheme) -> ColorTheme:
    colors = colors.model_copy()
    for name in colors.model_fields:
        val = getattr(colors, name)
        setattr(colors, name, to_web_color(val) or "")
    return colors
