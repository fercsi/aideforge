from datetime import datetime
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, Field

from aideforge.utils import IdType, id_to_timestamp

__all__ = [
    "BaseEvent",
    "BaseIncrementalStruct",
    "BaseStruct",
    "ChangeProperty",
    "PropertyChangeType",
]


class BaseStruct(BaseModel):
    """Base model for all persistent structs.

    A struct is a Pydantic model that can be stored and retrieved by a
    :class:`~aideforge.storage.abc.BaseStorage` backend.  It carries a
    ``kind`` and ``version`` along with optional meta fields that control
    storage scoping and naming.

    Attributes
    ----------
    kind : str
        Logical category or group of the struct.
    version : int
        Schema version of the struct.

    Meta information
    ----------------
    scope : ClassVar[Literal["global", "user"]]
        Storage scope that determines visibility.  ``"global"`` structs
        are shared across all users; ``"user"`` structs are isolated per
        user. Default is ``"user"``.
    storage_name : ClassVar[str]
        Format string used to derive the default storage name.  It is
        interpolated with the struct instance as the ``struct`` keyword
        argument.
    storage_mode : ClassVar[Literal["snapshot", "incremental"]]
        Mode how object is stored. In case of ``"snapshot"`` (default)
        object is always overwritten. If storage mode is ``"incremental"``
        the original object is always preserved, and changes are added
        incrementally.
    kinds : ClassVar[str or None]
        Sub-directory prefix for grouping related structs.  When ``None``,
        structs are stored directly under the root search path.
    """

    # model_config = ConfigDict(validate_assignment=True)

    kind: str
    version: int

    # Metadata
    scope: ClassVar[Literal["global", "user"]] = "user"
    storage_name: ClassVar[str] = "{struct.kind}"
    storage_mode: ClassVar[Literal["snapshot", "incremental"]] = "snapshot"
    kinds: ClassVar[str | None] = None

    @property
    def created_at(self) -> datetime:
        prop_name = f"{self.kind}_id"

        id = cast(IdType, getattr(self, prop_name, None))
        if id is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object does not support"
                f" 'created_at' because '{prop_name}' is missing."
            )

        return id_to_timestamp(self.event_id)


class BaseEvent(BaseStruct):
    # For event sourcing
    kind: Literal["event"] = "event"
    version: Literal[1] = 1
    event_id: IdType = Field(
        default_factory=lambda: create_id("event"),
        description="Unique identifier for the event",
    )
    event_type: str

    # metadata
    storage_name: ClassVar[str] = "{struct.event_id}"
    kinds: ClassVar[Literal["events"]] = "events"


class BaseIncrementalStruct(BaseStruct):
    events: list[BaseEvent] = Field(
        default_factory=list,
        description="Event list for event sourcing",
    )

    # Metadata
    storage_mode: ClassVar[Literal["incremental"]] = "incremental"


PropertyChangeType = Literal["replace", "append", "extend", "update", "add"]


class ChangeProperty(BaseEvent):
    event_type: Literal["change-property"] = "change-property"
    change_type: PropertyChangeType = Field(
        default="replace",
        description="The method how to change property.",
    )
    property_name: str = Field(..., description="Name of session property")
    old_value: Any = Field(default=None, description="Old value of property")
    new_value: Any = Field(..., description="New value of property")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._change_types = get_args(
            self.model_fields["change_type"].annotation
        )

    def change(self, struct: BaseIncrementalStruct):
        """Changes the value of a struct

        Use it to rebuild a structure.

        Parameters
        ----------
        struct : BaseIncrementalStruct
            The structure the property of which should be changed

        Example
        -------
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

        if change_type == "replace":
            setattr(struct, prop, self.new_value)
        elif change_type in self._change_types:
            # Dynamically call the corresponding method on current_value
            # (e.g., append, extend, update)
            getattr(current_value, change_type)(new_value)
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
        """Create event from changing a property

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

        Example
        -------
        >>> event = ChangeProperty(self.session, "current_agent", "12345678")
        >>> self.storage.store_struct_increment(self.session, event)
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
