from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Union, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from aideforge.exceptions import DuplicateKeyError
from .abc import BaseStruct
from .llm import LlmConfig
from .theme import Theme, theme_collection

if TYPE_CHECKING:
    from aideforge.abc.storage import AbstractStorage

__all__ = [
    "AssistantSettings",
    "BaseSettings",
    "SettingsGroup",
    "Switch",
    "ThinkingOption",
    "UiSettings",
    "get_settings_type",
    "serializable_settings",
    "validate_settings",
]

class Switch(StrEnum):
    OFF = "off"
    ON = "on"

class ThinkingOption(StrEnum):
    """Thinking options.

    Options
    -------
    OFF
        Don't show anything while thinking.
    TEXT
        Show the thinking text, but do not show tool calls.
    TOOL
        Don't show text but tool call information.
    ALL
        Show the thinking text and tool calls.
    DETAILS
        Show all information with details, like IDs
    """
    OFF = "off"
    TEXT = "text"
    TOOL = "tool"
    ALL = "all"
    DETAILS = "details"


class BaseSettings(BaseStruct):
    """Base class for all settings domains."""

    kind: Literal["settings"] = Field(
        default="settings",
        description="The type identifier, always \"settings\".",
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    domain: str = Field(
        ..., description="The settings domain name (e.g., \"assistant\", \"ui\")."
    )

    # Metadata
    storage_name: ClassVar[str] = "{struct.domain}"
    """*Model metadata*.
    Format string for storage name, interpolated with the struct instance.
    """
    kinds: ClassVar["settings"] = "settings"
    """*Model metadata*.
    Sub-directory for grouping settings.
    """
    type_fields: ClassVar[list[str]] = ["kind", "version", "domain"]
    """*Model metadata*.
    Field names which determine the structure type (non-data fields).
    """

    model_config = ConfigDict(validate_assignment=True)

    _storage: AbstractStorage | None = None

    def __init__(
        self, *, storage: AbstractStorage | None = None, **data: Any
    ) -> None:
        """Initialize settings with optional storage backend.

        Parameters
        ----------
        storage : AbstractStorage or None, optional
            Storage backend for persisting changes.
        **data : Any
            Field values for the settings.
        """
        super().__init__(**data)
        self._storage = storage

    def set_storage(self, storage: AbstractStorage) -> None:
        """Set the storage backend for this settings instance.

        Parameters
        ----------
        storage : AbstractStorage
            Storage backend to use for persistence.
        """
        self._storage = storage

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute and persist to storage if it's a model field.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        super().__setattr__(name, value)

        if name in type(self).model_fields:
            self._storage.store_struct(self)


class DisplaySettings(BaseModel):
    """Display and prompt configuration for the CLI."""

    prompt_string: str = Field(
        default="@[{agent}]> ",
        description="Template for the prompt string with {agent} and {model} placeholders.",
    )
    tool_prompt_string: str = Field(
        default=(
            "Permission required:\n"
            "  tool: {tool}\n"
            "  args: {args:pretty,8}\n"
            "y[es], n[o], a[lways], ...\n"
            "? "
        ),
        description="Template for tool permission prompts.",
    )
    tool_prompt_error_string: str = Field(
        default=(
            "y/yes, n/no, a/always, aat/always all tools, yfr/yes for run, "
            "yatfr/yes to all tool for run\n"
            "? "
        ),
        description="Template for tool permission error prompts.",
    )
    thinking: ThinkingOption = Field(
        default=ThinkingOption.OFF,
        description="Controls thinking output display mode (see :class:`ThinkingOption`).",
    )


class HistorySettings(BaseModel):
    """Configuration for the readline command history."""

    path: str | None = Field(
        default=None,
        description="File path to persist/read command history from. If None, history is not persisted.",
    )
    size: int | None = Field(
        default=1000,
        description="Maximum number of history entries to keep. None for infinite.",
    )


_settings_registry: dict[str, type[BaseSettings]] = {}


def serializable_settings(settings_type_name: str):
    """Decorator to register a new settings model from outside the module.

    Parameters
    ----------
    settings_type_name : str
        The domain name to register the settings class under.

    Returns
    -------
    Callable[[type[BaseSettings]], type[BaseSettings]]
        A decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[BaseSettings]) -> type[BaseSettings]:
        _settings_registry[settings_type_name] = cls
        return cls

    return decorator


def validate_settings(data: dict[str, Any]) -> BaseSettings:
    """Dynamically builds a discriminated union based on the current
    registry and validates the input data.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing settings data with a "domain" key.

    Returns
    -------
    BaseSettings
        Validated settings instance of the appropriate domain type.

    Raises
    ------
    ValueError
        If the domain is not registered.
    """

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
    """Retrieve the registered settings class for a domain.

    Parameters
    ----------
    domain : str
        The settings domain name.

    Returns
    -------
    type[BaseSettings]
        The registered settings class for the domain.

    Raises
    ------
    KeyError
        If the domain is not registered.
    """
    return _settings_registry[domain]


class SettingsGroup:
    r"""Container for managing multiple settings domains.

    Aggregates settings from different domains (e.g., assistant, ui) into
    a single accessible object. Handles registration, storage, and special
    key handlers for specific settings like themes.

    The individual settings or groups of settings are accessed (read and
    write) using subscription operations (``[]``). See *Examples* below.

    Parameters
    ----------
    storage : AbstractStorage or None, optional
        Storage backend for persisting settings.

    Examples
    --------
    >>> settings.register(UiSettings())
    >>> theme = settings["theme"]
    >>> ps = settings["display.prompt_string"]
    >>> settings["display", "prompt_string"] = "> "
    >>> display = settings["display"]
    >>> print(display.prompt_string)
    > 
    """

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
        """Set the storage backend for this settings group.

        Parameters
        ----------
        storage : AbstractStorage
            Storage backend to use for persistence.
        """
        self._storage = storage

    def register(
        self,
        settings: BaseSettings | dict[str, Any],
        override: bool = False,
        ignore_failure: bool = False,
    ) -> None:
        """Register a settings domain in this group.

        Parameters
        ----------
        settings : BaseSettings or dict[str, Any]
            Settings instance or dictionary to validate and register.
        override : bool, optional
            If True, replace existing domain. Default is False.
        ignore_failure : bool, optional
            If True, silently ignore duplicate domain. Default is False.

        Raises
        ------
        DuplicateKeyError
            If domain already exists and neither override nor ignore_failure.
        """
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
        """Check if a domain is registered.

        Parameters
        ----------
        domain : str
            Domain name to check.

        Returns
        -------
        bool
            True if domain is registered and is a BaseSettings instance.
        """
        return hasattr(self, domain) and isinstance(
            getattr(self, domain), BaseSettings
        )

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        """Get a setting value by key path.

        Parameters
        ----------
        key : str or tuple[str, ...]
            Setting key, optionally with nested path (e.g.,
            ``settings["history.size"]`` or ``settings["history", "size"]``).

        Returns
        -------
        Any
            The setting value.

        Raises
        ------
        KeyError
            If the key path doesn't exist.
        NotImplementedError
            If an intermediate type is not supported.
        """
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
        """Set a setting value by key path.

        Parameters
        ----------
        key : str or tuple[str, ...]
            Setting key, optionally with nested path. See also
            :meth:`__getitem__`.
        value : Any
            Value to set.

        Raises
        ------
        NotImplementedError
            If an intermediate type is not supported.
        """
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
        """Get a setting value with a default.

        Parameters
        ----------
        key : str
            Setting key.
        default : Any, optional
            Default value if key not found.

        Returns
        -------
        Any
            The setting value or default.
        """
        try:
            return self[key]
        except KeyError:
            return None

    def get_options(
        self, key: str | tuple[str, ...] | None = None
    ) -> list[str] | type:
        """Get available options for a setting key.

        Parameters
        ----------
        key : str or tuple[str, ...] or None, optional
            Setting key to get options for. If None, returns top-level keys.

        Returns
        -------
        list[str] or type
            List of option values, or type for non-enum settings.
        """
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
        """Get list of available theme names.

        Returns
        -------
        list[str]
            Sorted list of theme names from storage and built-in collection.
        """
        theme_names = self._storage.list_structs(Theme)
        names_set = set(self._themes.keys()) | {theme_names}
        return list(sorted(names_set))

    def set_theme(self, theme_name: str) -> Exception | None:
        """Load and register a theme by name.

        Parameters
        ----------
        theme_name : str
            Name of the theme to load.

        Returns
        -------
        Exception or None
            Exception if theme not found, None on success.
        """
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
        """Get a theme by name or the current UI theme.

        Parameters
        ----------
        theme_name : str or None, optional
            Theme name to retrieve. If None, uses current UI theme.

        Returns
        -------
        Theme
            The requested theme.
        """
        return self._themes[theme_name or self.ui.theme]


# Default Settings classes
@serializable_settings("assistant")
class AssistantSettings(BaseSettings):
    """Settings for the AI assistant behavior."""

    domain: Literal["assistant"] = Field(
        default="assistant",
        description="The settings domain, always \"assistant\".",
    )
    history: HistorySettings = Field(
        default_factory=HistorySettings,
        description="Command history configuration.",
    )
    llm: LlmConfig = Field(
        ..., description="Default LLM for Assistant"
    )


@serializable_settings("ui")
class UiSettings(BaseSettings):
    """Settings for the user interface."""

    domain: Literal["ui"] = Field(
        default="ui",
        description="The settings domain, always \"ui\".",
    )
    theme: str = Field(
        default="default",
        description="Name of the active theme.",
    )
    display: DisplaySettings = Field(
        default_factory=DisplaySettings,
        description="Display and prompt configuration.",
    )
