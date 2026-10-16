from typing import ClassVar, Literal

from pydantic import BaseModel

__all__ = ["BaseStruct"]


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

    Class Variables
    ---------------
    scope : ClassVar[Literal["global", "user"]]
        Storage scope that determines visibility.  ``"global"`` structs
        are shared across all users; ``"user"`` structs are isolated per
        user.
    storage_name : ClassVar[str]
        Format string used to derive the default storage name.  It is
        interpolated with the struct instance as the ``struct`` keyword
        argument.
    kinds : ClassVar[str or None]
        Sub-directory prefix for grouping related structs.  When ``None``,
        structs are stored directly under the root search path.
    """

    kind: str
    version: int

    # meta fields
    scope: ClassVar[Literal["global", "user"]] = "user"
    storage_name: ClassVar[str] = "{struct.kind}"
    kinds: ClassVar[str | None] = None
