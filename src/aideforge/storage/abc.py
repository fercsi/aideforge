from abc import ABC, abstractmethod
from typing import TypeVar

from aideforge.structs.abc import BaseStruct, BaseIncrementalStruct, BaseEvent

__all__ = ["BaseStorage", "StructT"]

StructT = TypeVar("StructT", bound=BaseStruct)


class BaseStorage(ABC):
    """Abstract base class for struct storage backends.

    A storage backend is responsible for persisting, retrieving, and
    listing struct instances (``BaseStruct`` subclasses) in a
    concrete medium (e.g. the filesystem, a database, or a remote API).
    """

    @abstractmethod
    def retrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failures: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the storage backend.

        Parameters
        ----------
        struct_type : type[StructT]
            The BaseStruct class of the struct to retrieve.
        struct_name : str or None
            Name of the struct to retrieve.  When ``None``, a default
            name is derived from ``struct_type.storage_name``.
        ignore_failures : bool
            When ``True``, return ``None`` instead of raising if the
            struct cannot be found.

        Returns
        -------
        StructT or None
            The retrieved and validated struct, or ``None`` if not found
            and ``ignore_failures`` is ``True``.
        """
        pass

    @abstractmethod
    def store_struct(
        self, struct: BaseStruct, struct_name: str | None = None
    ) -> None:
        """Store a struct in the storage backend.

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to store.
        struct_name : str or None
            Name under which the struct is stored.  When ``None``, a
            default name is derived from ``struct.storage_name``.
        """
        pass

    @abstractmethod
    def store_struct_increment(
        self,
        struct: BaseIncrementalStruct,
        increment: BaseEvent,
        *,
        struct_name: str | None = None,
    ) -> None:
        """Stores an increment to an incrementally stored struct

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to store.
        increment : BaseStruct
            The BaseStruct increment instance to store within ``struct`'s
            scope.
        struct_name : str or None
            Name under which the struct is stored.  When ``None``, a
            default name is derived from ``struct.storage_name``.
        """
        pass

    @abstractmethod
    def list_structs(self, struct_type: type[BaseStruct]) -> list[str]:
        """List all stored struct names for a given struct type.

        Parameters
        ----------
        struct_type : type[BaseStruct]
            The BaseDtruct class whose stored instances should be listed.

        Returns
        -------
        list[str]
            Sorted list of struct names found in the storage backend.
        """
        pass
