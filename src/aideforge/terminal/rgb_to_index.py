from functools import lru_cache
from typing import NamedTuple, Sequence, overload

try:
    import rich
    from rich.color import Color, ColorSystem, ColorType

    RICH_SUPPORTED = True
except:
    RICH_SUPPORTED = False

__all__ = ["web_to_xterm_index", "rgb_to_xterm_index"]

_LIGHTNESS_SCALING: float = 2.0


class RgbColor(NamedTuple):
    r: int
    g: int
    b: int


class XyzColor(NamedTuple):
    x: float
    y: float
    z: float


class LuvColor(NamedTuple):
    L: float
    u: float
    v: float


@lru_cache(maxsize=1024)
def web_to_xterm_index(hex_color: str) -> int:
    """Convert web hexadecimal color format to xterm-256color palette."""
    rgb = _hex_to_rgb(hex_color)
    return rgb_to_xterm_index(rgb)


@lru_cache(maxsize=1024)
def rgb_to_xterm_index(color: tuple[int, int, int]) -> int:
    """Convert an RGB color to xterm-256color palette."""
    return _rgb_to_index(color, luv_palette=_XTERM_COLORS_LUV)


def _rgb_to_index(
    color: tuple[int, int, int],
    luv_palette: Sequence[tuple[float, float, float]],
) -> int:
    """Convert an RGB color to xterm-256color palette."""
    color = RgbColor(*color)

    trg_L, trg_u, trg_v = _rgb_to_luv(color)

    trg_L *= _LIGHTNESS_SCALING

    min_dist = float("inf")
    best_idx = 0

    for idx, (L, u, v) in enumerate(luv_palette):
        if idx < 16:
            # don't use system colors, because they are usually adjusted
            continue
        L_weighted = L * _LIGHTNESS_SCALING
        dist = (L_weighted - trg_L) ** 2 + (u - trg_u) ** 2 + (v - trg_v) ** 2
        if dist < min_dist:
            min_dist = dist
            best_idx = idx

    return best_idx


def _hex_to_rgb(hex_color: str) -> RgbColor:
    """Convert web hexadecimal color format to RGB."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters")
    return RgbColor(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_xyz(color: RgbColor) -> XyzColor:
    """Transform an RGB color to XYZ color space."""

    def srgb_to_linear(c: float) -> float:
        c = c / 255.0
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    # Linearize RGB
    r = srgb_to_linear(color.r)
    g = srgb_to_linear(color.g)
    b = srgb_to_linear(color.b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return XyzColor(x, y, z)


def _xyz_to_luv(color: XyzColor) -> LuvColor:
    r"""Transform an XYZ color to Lu\*v\* color space."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    x, y, z = color

    if y == 0:
        return LuvColor(0.0, 0.0, 0.0)

    u_prime = 4 * x / (x + 15 * y + 3 * z)
    v_prime = 9 * y / (x + 15 * y + 3 * z)

    u_n = 4 * Xn / (Xn + 15 * Yn + 3 * Zn)
    v_n = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)

    y_ratio = y / Yn
    if y_ratio > 0.008856:
        L = 116 * (y_ratio ** (1 / 3)) - 16
    else:
        L = 903.3 * y_ratio

    u = 13 * L * (u_prime - u_n)
    v = 13 * L * (v_prime - v_n)

    return LuvColor(L, u, v)


def _rgb_to_luv(color: RgbColor) -> LuvColor:
    r"""Transform an RGB color to Lu\*v\* color space."""
    xyz_color = _rgb_to_xyz(color)
    return _xyz_to_luv(xyz_color)


def _generate_indices() -> list[tuple[int, int, int]]:
    colors = []

    # System colors (16 color palette)
    for i in range(16):
        v = 255 if i > 8 else 128
        colors.append(RgbColor(i & 1 and v, i & 2 and v, i & 4 and v))
    colors[7:9] = (RgbColor(192, 192, 192), RgbColor(128, 128, 128))

    # RGB color cube (16-231)
    values = [0, 95, 135, 175, 215, 255]
    for r in values:
        for g in values:
            for b in values:
                colors.append(RgbColor(r, g, b))

    # Grayscale ramp (232-255)
    for i in range(24):
        v = 8 + i * 10
        colors.append(RgbColor(v, v, v))

    return colors


_XTERM_COLORS_RGB = _generate_indices()
_XTERM_COLORS_LUV = [_rgb_to_luv(rgb) for rgb in _XTERM_COLORS_RGB]


if RICH_SUPPORTED:
    __all__.extend(["patch_rich", "unpatch_rich"])

    _original_downgrade = None
    _original_match = None

    def patch_rich():
        global _original_downgrade, _original_match
        if _original_downgrade is not None:
            return
        print("!!!!!")

        _original_downgrade = rich.color.Color.downgrade
        rich.color.Color.downgrade = patched_downgrade

        _original_match = rich.palette.Palette.match
        rich.palette.Palette.match = patched_match
        rich.color.WINDOWS_PALETTE.match = patched_match
        rich.color.STANDARD_PALETTE.match = patched_match
        rich.color.EIGHT_BIT_PALETTE.match = patched_match

    def unpatch_rich():
        global _original_downgrade, _original_match
        if _original_downgrade is None:
            return

        rich.color.Color.downgrade = _original_downgrade
        _original_downgrade = None

        rich.palette.Palette.match = _original_match
        rich.color.WINDOWS_PALETTE.match = _original_match
        rich.color.STANDARD_PALETTE.match = _original_match
        rich.color.EIGHT_BIT_PALETTE.match = _original_match
        _original_match = None

    @lru_cache(maxsize=1024)
    def patched_downgrade(self, system: ColorSystem) -> Color:
        # Convert to 8-bit color from truecolor color
        global original_downgrade
        if (
            system == ColorSystem.EIGHT_BIT
            and self.system == ColorSystem.TRUECOLOR
        ):
            assert self.triplet is not None
            index = rgb_to_xterm_index(self.triplet)
            return Color(self.name, ColorType.EIGHT_BIT, number=index)

        return _original_downgrade(self, system)

    @lru_cache(maxsize=1024)
    def patched_match(self, color: tuple[int, int, int]) -> int:
        if not hasattr(self, "_luv_colors"):
            self._luv_colors = tuple(
                _rgb_to_luv(RgbColor(*rgb)) for rgb in self._colors
            )
        return _rgb_to_index(color, luv_palette=self._luv_colors)
