from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .common import Switch
from .llm import LlmConfig

__all__ = [
    "AssistantSettings",
    "BaseSettings",
    "UiSettings",
    "get_settings_type",
    "validate_settings",
]


class ThinkingOption(StrEnum):
    OFF = "off"
    TEXT = "text"
    ALL = "all"
    DETAILS = "details"


class BaseSettings(BaseModel):
    kind: Literal["settings"] = "settings"
    version: Literal[1] = 1
    scope: str

    # Metadata
    storage_name: ClassVar[str] = "{struct.scope}"
    kinds: ClassVar["settings"] = "settings"


class DisplaySettings(BaseModel):
    """
    prompt_string : str or None, optional
        A template for the prompt string. May contain placeholders such as
        ``{agent}`` and ``{model}``.
    """

    prompt_string: str = "@[{agent}]> "
    tool_prompt_string: str = (
        "Permission required:\n"
        "  tool: {tool}\n"
        "  args: {args:pretty,8}\n"
        "y[es], n[o], a[lways], ...\n"
        "? "
    )
    tool_prompt_error_string: str = (
        "y/yes, n/no, a/always, aat/always all tools, yfr/yes for run, "
        "yatfr/yes to all tool for run\n"
        "? "
    )
    thinking: ThinkingOption = ThinkingOption.OFF

class HistorySettings(BaseModel):
    """Configuration for the readline command history.

    Attributes
    ----------
    path : str or None, optional
        File path to persist/read command history from. If ``None``,
        history is not persisted to disk.
    size : int or None, optional
        Maximum number of history entries to keep. If ``None``, the
        readline default is used.
    """

    path: str | None = None
    size: int | None = 1000 # None: infinite


_settings_registry: dict[str, type[BaseSettings]] = {}


def register_settings(settings_type_name: str):
    """Decorator to register a new settings model from outside the module."""

    def decorator(cls: type[BaseSettings]) -> type[BaseSettings]:
        _settings_registry[settings_type_name] = cls
        return cls

    return decorator


# Default Settings classes
@register_settings("assistant")
class AssistantSettings(BaseSettings):
    scope: Literal["assistant"] = "assistant"
    history: HistorySettings = Field(default_factory=HistorySettings)
    llm: LlmConfig = Field(..., description="Default LLM for Assistant")


@register_settings("dog")
class UiSettings(BaseSettings):
    scope: Literal["ui"] = "ui"
    theme: str = "default"
    display: DisplaySettings = Field(default_factory=DisplaySettings)


def validate_settings(data: dict[str, Any]) -> BaseSettings:
    """Dynamically builds a discriminated union based on the current
    registry and validates the input data."""

    # Verify that the incoming scope is known
    scope_key = data.get("scope")
    if scope_key not in _settings_registry:
        raise ValueError(f"Unknown settings scope: {scope_key}")

    # Combine all registered models into a Union tuple
    union_type = Union[tuple(_settings_registry.values())]

    # Create a TypeAdapter with the discriminated union and validate
    adapter = TypeAdapter(Annotated[union_type, Field(discriminator="scope")])
    return adapter.validate_python(data)


def get_settings_type(scope: str) -> BaseSettings:
    return _settings_registry[scope]
