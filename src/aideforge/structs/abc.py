from datetime import datetime
from typing import Any, ClassVar, Literal, cast, get_args

from pydantic import BaseModel, Field

from aideforge.utils import IdType, id_to_timestamp, create_id

__all__ = [
    "BaseEvent",
    "BaseIncrementalStruct",
    "BaseStruct",
    "ChangeProperty",
    "PropertyChangeType",
    "StructProxy",
]


class BaseStruct(BaseModel):
    """Base model for all persistent structs.

    A struct is a Pydantic model that can be stored and retrieved by a
    :class:`~aideforge.abc.storage.AbstractStorage` backend.  It carries a
    ``kind`` and ``version`` along with optional meta fields that control
    storage scoping and naming.
    """

    # model_config = ConfigDict(validate_assignment=True)

    kind: str = Field(
        ...,
        description="Logical category or group of the struct.",
    )
    version: int = Field(
        ...,
        description="Schema version of the struct kind.",
    )

    # Metadata
    scope: ClassVar[Literal["global", "user"]] = "user"
    """*Model metadata*.
    Storage scope that determines visibility.  ``"global"`` structs
    are shared across all users; ``"user"`` structs are isolated per
    user. Default is ``"user"``.
    """

    storage_name: ClassVar[str] = "{struct.kind}"
    """*Model metadata*.
    Format string used to derive the default storage name.  It is
    interpolated with the struct instance as the ``struct`` keyword
    argument.
    """

    storage_mode: ClassVar[Literal["snapshot", "incremental"]] = "snapshot"
    """*Model metadata*.
    Mode how object is stored. In case of ``"snapshot"`` (default)
    object is always overwritten. If storage mode is ``"incremental"``
    the original object is always preserved, and changes are added
    incrementally.
    """

    kinds: ClassVar[str | None] = None
    """*Model metadata*.
    Sub-directory prefix for grouping related structs.  When ``None``,
    structs are stored directly under the root search path.
    """

    type_fields: ClassVar[list[str]] = ["kind", "version"]
    """*Model metadata*.
    Contains field names which determine the structure type (non-data fields).
    """

    @property
    def created_at(self) -> datetime:
        """Return struct's creastion time.

        AideForge IDs contain a timestamp. This property evaluates and returns
        this timestamp in ``datetime.datetime`` format.

        .. note::
           This property is available only with structs having their own ID
           of form ``<kind>_id``.
        """
        prop_name = f"{self.kind}_id"

        id = cast(IdType, getattr(self, prop_name, None))
        if id is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object does not support"
                f" 'created_at' because '{prop_name}' is missing."
            )

        return id_to_timestamp(id)


class BaseEvent(BaseStruct):
    """Base model for all event sourcing items."""

    # For event sourcing
    kind: Literal["event"] = "event"
    version: Literal[1] = 1
    event_id: IdType = Field(
        default_factory=lambda: create_id("event"),
        description="Unique identifier for the event",
    )
    event_type: str = Field(
        ...,
        description=(
            "Type of event item. Every event types has their own inherited "
            "event struct"
        ),
    )

    # metadata
    storage_name: ClassVar[str] = "{struct.event_id}"
    kinds: ClassVar[Literal["events"]] = "events"


class BaseIncrementalStruct(BaseStruct):
    """Base model for all incremental persistent structs.

    ``BaseIncrementalStruct`` structures are designed to hold continuously
    growing information and events. Regularly saving these structures
    becomes increasingly resource-intensive over time. Structures based on
    this model contain initial/default values, while changes are stored
    incrementally as events (``BaseEvent``) only.

    This makes it possible to update the structure's contents using append
    mode, even within a file.
    """

    events: list[BaseEvent] = Field(
        default_factory=list,
        description=(
            "List of events including the change property events. See also "
            ":class:`ChangeProperty`."
        ),
    )

    # Metadata
    storage_mode: ClassVar[Literal["incremental"]] = "incremental"


PROPERTY_CHANGE_METHODS = ("replace", "append", "extend", "update", "add")
PropertyChangeType = Literal[*PROPERTY_CHANGE_METHODS]


class ChangeProperty(BaseEvent):
    """Change property event for incremental storage

    An event that captures changes to the parameters of structures based on
    the :class:`BaseIncrementalStruct` model.

    The event can be used in two ways.

    The first is when changing a property. In this case, instead of
    simply assigning a new value, we modify the structure using ``event
    = ChangeProperty.from_changing(struct, property_name, new_value)``.
    The resulting event is then stored using the storage object's
    :meth:`~aideforge.abc.storage.AbstractStorage.store_struct_increment`
    method. See :meth:`from_changing`

    The second is when reading the structure
    (``change_property_event.change(struct)``). We load the original
    structure and then replay the events one by one to reconstruct its
    current state. See :meth:`change`
    """

    event_type: Literal["change-property"] = "change-property"
    change_type: PropertyChangeType = Field(
        default="replace",
        description=(
            'The method how to change the property. Defaults to ``"replace"``. '
            'Possible options: ``"replace"``, ``"append"``, ``"extend"``, '
            '``"update"``, ``"add"``.'
        ),
    )
    property_name: str = Field(..., description="Name of session property")
    old_value: Any = Field(default=None, description="Old value of property")
    new_value: Any = Field(..., description="New value of property")

    def change(self, struct: BaseIncrementalStruct):
        r"""Changes the value of a struct

        Use it to rebuild a structure.

        Parameters
        ----------
        struct : BaseIncrementalStruct
            The structure the property of which should be changed

        Examples
        --------
        >>> struct = original
        ... for event in event_list:
        ...   if event.type != "change-property":
        ...     continue
        ...   chprop = cast(ChangeProperty, event)
        ...   chprop.change(struct)
        """
        prop = self.property_name
        current_value = getattr(
            struct, prop
        )  # raises exception if property doesn't exist

        if self.change_type == "replace":
            setattr(struct, prop, self.new_value)
        elif self.change_type in PROPERTY_CHANGE_METHODS:
            # Dynamically call the corresponding method on current_value
            # (e.g., append, extend, update)
            getattr(current_value, self.change_type)(self.new_value)
        else:
            raise ValueError(f"Unsupported change type: {self.change_type}")

    @classmethod
    def from_changing(
        cls,
        struct: BaseIncrementalStruct,
        property_name: str,
        new_value: Any,
        *,
        change_type: PropertyChangeType = "replace",
    ):
        r"""Create event from changing a property

        This class method changes the property, including adding the event to
        the ``events`` list and returns the event, so the caller can send it
        to the storage component.

        Parameters
        ----------
        struct : BaseIncrementalStruct
            The structure the property of which should be changed
        property_name : str
            The name of the property to change
        new_value : Any
            The new value to replace with, to append, etc.
        change_type : PropertyChangeType, optional
            The method how to change the property. Defaults to ``"replace"``.
            Possible options: ``"replace"``, ``"append"``, ``"extend"``,
            ``"update"``, ``"add"``.

        Examples
        --------
        >>> event = ChangeProperty(session, "current_agent", "12345678")
        >>> storage.store_struct_increment(session, event)
        """
        old_value = getattr(struct, property_name)  # raises exception
        event = ChangeProperty(
            change_type=change_type,
            property_name=property_name,
            old_value=old_value,
            new_value=new_value,
        )
        event.change(struct)
        struct.events.append(event)
        return event


class StructProxy:
    """Internal proxy that makes a ``BaseStruct`` subclass behave like an
    instance of its default values.

    For smaller structures, an object can be instantiated directly, but for
    larger ones, using a proxy is particularly useful.

    Parameters
    ----------
    target : type[BaseStruct]
        The :class:`BaseStruct` subclass type to proxy.

    Examples
    --------
    >>> class MyStruct(BaseStruct):
    ...   kind : Literal["my-struct"] = "my-struct"
    ...
    >>> MyStruct.kind
    Traceback (most recent call last):
    ...
    AttributeError: kind
    >>> proxy = StructProxy(MyStruct)
    >>> proxy.kind
    'my-struct'
    """

    def __init__(self, target: type[BaseStruct]) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Any:
        """Retrieve an attribute from the proxied class.

        Parameters
        ----------
        name : str
            Attribute name to retrieve.

        Returns
        -------
        Any
            The class-level attribute value (for meta parameters) or the
            default value of the matching structure field.
        """
        if hasattr(self._target, name):  # meta parameter
            return getattr(self._target, name)
        field_info = self._target.model_fields.get(name)
        if field_info is None:
            raise AttributeError(
                f"'{self._target.__name__}' object has no attribute '{name}'"
            )
        return field_info.default
