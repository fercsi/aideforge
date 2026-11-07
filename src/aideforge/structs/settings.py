from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Union, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from aideforge.exceptions import DuplicateKeyError
from .abc import BaseStruct
from .common import Switch
from .llm import LlmConfig
from .theme import Theme, theme_collection

if TYPE_CHECKING:
    from aideforge.abc.storage import AbstractStorage

__all__ = [
    "AssistantSettings",
    "BaseSettings",
    "SettingsGroup",
    "UiSettings",
    "get_settings_type",
    "serializable_settings",
    "validate_settings",
]


class ThinkingOption(StrEnum):
    OFF = "off"
    TEXT = "text"
    ALL = "all"
    DETAILS = "details"


class BaseSettings(BaseStruct):
    kind: Literal["settings"] = "settings"
    version: Literal[1] = 1
    domain: str

    # Metadata
    storage_name: ClassVar[str] = "{struct.domain}"
    kinds: ClassVar["settings"] = "settings"
    type_fields: ClassVar[list[str]] = ["kind", "version", "domain"]

    model_config = ConfigDict(validate_assignment=True)

    _storage: AbstractStorage | None = None

    def __init__(
        self, *, storage: AbstractStorage | None = None, **data: Any
    ) -> None:
        super().__init__(**data)
        self._storage = storage

    def set_storage(self, storage: AbstractStorage) -> None:
        self._storage = storage

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name in type(self).model_fields:
            self._storage.store_struct(self)


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
    size: int | None = 1000  # None: infinite


_settings_registry: dict[str, type[BaseSettings]] = {}


def serializable_settings(settings_type_name: str):
    """Decorator to register a new settings model from outside the module."""

    def decorator(cls: type[BaseSettings]) -> type[BaseSettings]:
        _settings_registry[settings_type_name] = cls
        return cls

    return decorator


def validate_settings(data: dict[str, Any]) -> BaseSettings:
    """Dynamically builds a discriminated union based on the current
    registry and validates the input data."""

    # Verify that the incoming domain is known
    domain_key = data.get("domain")
    if domain_key not in _settings_registry:
        raise ValueError(f"Unknown settings domain: {domain_key}")

    # Combine all registered models into a Union tuple
    union_type = Union[tuple(_settings_registry.values())]

    # Create a TypeAdapter with the discriminated union and validate
    adapter = TypeAdapter(Annotated[union_type, Field(discriminator="domain")])
    return adapter.validate_python(data)


def get_settings_type(domain: str) -> BaseSettings:
    return _settings_registry[domain]


class SettingsGroup:
    _settings: dict[str, BaseSettings]

    def __init__(self, storage: AbstractStorage | None = None) -> None:
        self._settings = {}
        self._storage = storage
        self._spec_opt_handlers = {
            ("theme"): self.get_themes,
        }
        self._spec_set_handlers = {
            ("theme"): self.set_theme,
        }
        self._spec_get_handlers = {
            ("theme"): self.get_theme,
        }
        self._themes = theme_collection.copy()

    def set_storage(self, storage: AbstractStorage) -> None:
        self._storage = storage

    def register(
        self,
        settings: BaseSettings | dict[str, Any],
        override: bool = False,
        ignore_failure: bool = False,
    ) -> None:
        if isinstance(settings, dict):
            settings = validate_settings(settings)
        name = settings.domain
        if not hasattr(self, name) or override:
            settings.set_storage(self._storage)
            for prim in settings.model_fields:
                if prim in settings.type_fields:
                    continue
                self._settings[prim] = settings
                self[prim] = getattr(settings, prim)  # run spechandlers
            setattr(self, name, settings)
        elif ignore_failure:
            return None
        else:
            raise DuplicateKeyError(
                f"Multiple settings has been given with the same domain: '{name}'"
            )

    def has_domain(self, domain) -> bool:
        return hasattr(self, domain) and isinstance(
            getattr(self, domain), BaseSettings
        )

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        if isinstance(key, str):
            key = (key,)
        keys = [k for p in key for k in p.split(".")]

        item = self._settings[keys[0]]
        for k in keys:
            if isinstance(item, BaseModel):
                item = getattr(item, k)
            elif isinstance(item, dict):
                item = item.get(k)
            else:
                raise NotImplementedError(
                    f"Type '{type(item).__name__}' not supported"
                )
        return item

    def __setitem__(self, key: str | tuple[str, ...], value: Any) -> None:
        if isinstance(key, str):
            key = (key,)
        keys = tuple(k for p in key for k in p.split("."))
        spechandler = self._spec_set_handlers.get(keys)
        if spechandler is not None:
            # may raise ValueError and then setting does not change
            spechandler(value)

        item = self._settings[keys[0]]
        for k in keys[:-1]:
            if isinstance(item, BaseModel):
                item = getattr(item, k)
            elif isinstance(item, dict):
                item = item.get(k)
            else:
                raise NotImplementedError(
                    f"Type '{type(item).__name__}' not supported"
                )
        if isinstance(item, BaseModel):
            # TODO: check not to override model -> pydantic?!
            setattr(item, keys[-1], value)
        elif isinstance(item, dict):
            item[keys[-1]] = value
        else:
            raise NotImplementedError(
                f"Type '{type(item).__name__}' not supported"
            )
        spechandler = self._spec_set_handlers.get(keys)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def get_options(
        self, key: str | tuple[str, ...] | None = None
    ) -> list[str] | type:
        # todo: limitations, like >=0, length <= 100...
        if not key:  # None or empty szring
            return list(self._settings.keys())
        if isinstance(key, str):
            key = (key,)
        keys = tuple(k for p in key for k in p.split("."))
        spechandler = self._spec_opt_handlers.get(keys)
        if spechandler is not None:
            return spechandler()

        item = self._settings[keys[0]]
        for k in keys:
            if isinstance(item, BaseModel):
                item = getattr(item, k)
            elif isinstance(item, dict):
                item = item.get(k)
            else:
                raise NotImplementedError(
                    f"Type '{type(item).__name__}' not supported"
                )
        if isinstance(item, BaseSettings):
            return list(set(item.model_fields) - set(item.type_fields))
        elif isinstance(item, BaseModel):
            return item.model_fields.copy()
        elif isinstance(item, dict):
            return list(item.keys())
        elif isinstance(item, StrEnum):
            return [e.value for e in type(item)]
        else:
            return type(item)

    def get_themes(self) -> list[str]:
        theme_names = self._storage.list_structs(Theme)
        names_set = set(self._themes.keys()) | {theme_names}
        return list(sorted(names_set))

    def set_theme(self, theme_name: str) -> Exception | None:
        if self._storage is None:
            return
        if theme_name in self._themes:
            return
        theme = self._storage.retrieve_struct(
            Theme, theme_name, ignore_errors=True
        )
        if theme is None:
            raise ValueError("Theme '{theme_name}' not found")
        self._themes[theme_name] = theme
        return

    def get_theme(self, theme_name: str | None = None) -> Theme:
        return self._themes[theme_name or self.ui.theme]


# Default Settings classes
@serializable_settings("assistant")
class AssistantSettings(BaseSettings):
    domain: Literal["assistant"] = "assistant"
    history: HistorySettings = Field(default_factory=HistorySettings)
    llm: LlmConfig = Field(..., description="Default LLM for Assistant")


@serializable_settings("ui")
class UiSettings(BaseSettings):
    domain: Literal["ui"] = "ui"
    theme: str = "default"
    display: DisplaySettings = Field(default_factory=DisplaySettings)
