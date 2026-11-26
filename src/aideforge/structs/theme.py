from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .abc import BaseStruct

__all__ = [
    "DecorationType",
    "TextStyle",
    "TextStyles",
    "Theme",
    "theme_collection",
]


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
    text_styles : TextStyles
        The text styles to apply. Defaults to a :class:`TextStyles` instance
        with all styles set to ``None`` (monochrome).
    """

    kind: Literal["theme"] = "theme"
    version: Literal[1] = 1
    name: str = Field(..., description="Name of the theme to identify with")
    text_styles: TextStyles = Field(
        default_factory=TextStyles,
        description="",
    )

    # Metadata
    scope: ClassVar[Literal["global"]] = "global"
    storage_name: ClassVar[str] = "{struct.name}"
    kinds: ClassVar[Literal["themes"]] = "themes"


theme_collection: dict[str, Theme] = {
    "default": Theme(
        name="default",
        text_styles={
            "welcome_message": {"color": "bright_yellow"},
            "tool_prompt_string": {"color": "bright_cyan"},
            "llm_id": {"color": "white"},
            "llm_thinking": {"color": "bright_black"},
            "llm_tool_call": {"color": "cyan"},
            "llm_tool_output": {"color": "cyan"},
            "response": {"color": "bright_yellow"},
            "output_error": {"color": "bright_white"},
            "error": {"color": "bright_red"},
        },
    ),
    "monochrome": Theme(
        name="monochrome",
        text_styles={},
    ),
}
