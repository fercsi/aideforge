from abc import ABC, abstractmethod
from typing import TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver

from aideforge.structs import BaseStruct, BaseIncrementalStruct, BaseEvent

__all__ = ["AbstractStorage", "StructT"]

StructT = TypeVar("StructT", bound=BaseStruct)


class AbstractStorage(ABC):
    """Abstract base class for struct storage backends.

    A storage backend is responsible for persisting, retrieving, and
    listing struct instances (``BaseStruct`` subclasses) in a
    concrete medium (e.g. the filesystem, a database, or a remote API).

    NOTE: Every function has both a synchronous and an asynchronous version.
    By default, each asynchronous version runs the corresponding synchronous
    function in a separate thread using ``asyncio.to_thread()``. In many
    cases, this is sufficient, and there is no need to implement a dedicated
    asynchronous version.
    """

    def prepare(self) -> None:
        """Prepare storage operations
        """
        pass

    async def aprepare(self) -> None:
        """Prepare storage operations

        This is the asynchron version of :meth:`prepare`.
        """
        return await asyncio.to_thread(self.prepare)
        
    def cleanup(self) -> None:
        """Clean up storage operations
        """
        pass

    async def acleanup(self) -> None:
        """Clean up storage operations

        This is the asynchron version of :meth:`cleanup`.
        """
        return await asyncio.to_thread(self.cleanup)

    @abstractmethod
    def retrieve_resource(
        self,
        path: str,
        *,
        mode: str = "t",
        ignore_failure: bool = False,
    ) -> str | bytes | None:
        """Retrieve a resource from the storage backend.

        Parameters
        ----------
        path : str
            Path of the resource to retrieve.
        mode : str, optional
            Mode how the resource should be returned, ``"t"`` for text (default)
            and ``"b"`` for binary.
        ignore_failure : bool, optional
            When ``True``, return ``None`` instead of raising if the
            resource cannot be found.

        Returns
        -------
        str or bytes or None
            The retrieved content, or ``None`` if not found
            and ``ignore_failure`` is ``True``.
        """
        pass

    async def aretrieve_resource(
        self,
        path: str,
        *,
        mode: str = "t",
        ignore_failure: bool = False,
    ) -> str | bytes | None:
        """Retrieve a resource from the storage backend.

        This is the asynchron version of :meth:`retrieve_resource`.
        """
        return await asyncio.to_thread(
            self.retrieve_resource,
            path=path,
            mode=mode,
            ignore_failure=ignore_failure,
        )

    @abstractmethod
    def store_resource(
        self,
        path: str,
        content: str | bytes,
        *,
        ignore_failure: bool = False,
    ) -> None:
        """Store a resource to the storage backend.

        Parameters
        ----------
        path : str
            Path of the resource to retrieve.
        content : str | bytes
            Content of the resource to be stored.
        ignore_failure : bool
            When ``True``, return instead of raising if the
            resource cannot be stored.
        """
        pass

    async def astore_resource(
        self,
        path: str,
        content: str | bytes,
        *,
        ignore_failure: bool = False,
    ) -> None:
        """Store a resource to the storage backend.

        This is the asynchron version of :meth:`store_resource`.
        """
        return await asyncio.to_thread(
            self.store_resource,
            path=path,
            content=content,
            ignore_failure=ignore_failure,
        )

    @abstractmethod
    def retrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the storage backend.

        Parameters
        ----------
        struct_type : type[StructT]
            The BaseStruct class of the struct to retrieve.
        struct_name : str or None
            Name of the struct to retrieve.  When ``None``, a default
            name is derived from ``struct_type.storage_name``.
        ignore_failure : bool
            When ``True``, return ``None`` instead of raising if the
            struct cannot be found.

        Returns
        -------
        StructT or None
            The retrieved and validated struct, or ``None`` if not found
            and ``ignore_failure`` is ``True``.
        """
        pass

    async def aretrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the storage backend.

        This is the asynchron version of :meth:`retrieve_struct`.
        """
        return await asyncio.to_thread(
            self.retrieve_struct,
            struct_type=struct_type,
            struct_name=struct_name,
            ignore_failure=ignore_failure,
        )

    @abstractmethod
    def store_struct(
        self,
        struct: BaseStruct,
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
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

    async def astore_struct(
        self,
        struct: BaseStruct,
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> None:
        """Store a struct in the storage backend.

        This is the asynchron version of :meth:`store_struct`.
        """
        return await asyncio.to_thread(
            self.store_struct,
            struct_type=struct_type,
            struct_name=struct_name,
            ignore_failure=ignore_failure,
        )

    @abstractmethod
    def store_struct_increment(
        self,
        struct: BaseIncrementalStruct,
        increment: BaseEvent,
        *,
        struct_name: str | None = None,
        ignore_failure: bool = False,
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

    async def astore_struct_increment(
        self,
        struct: BaseIncrementalStruct,
        increment: BaseEvent,
        *,
        struct_name: str | None = None,
        ignore_failure: bool = False,
    ) -> None:
        """Stores an increment to an incrementally stored struct

        This is the asynchron version of :meth:`store_struct_increment`.
        """
        return await asyncio.to_thread(
            self.store_struct_increment,
            struct_type=struct_type,
            increment=increment,
            struct_name=struct_name,
            ignore_failure=ignore_failure,
        )

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

    async def alist_structs(self, struct_type: type[BaseStruct]) -> list[str]:
        """List all stored struct names for a given struct type.

        This is the asynchron version of :meth:`list_structs`.
        """
        return await asyncio.to_thread(
            self.list_structs,
            struct_type=struct_type,
        )

    @abstractmethod
    def get_checkpoint_saver(
        self, saver_id: str | None = None
    ) -> BaseCheckpointSaver:
        """Returns the requested checkpoint saver

        Usually a single checkpoint saver for the whole application is
        sufficient. If multiple savers are needed, use ``saver_id`` to
        identify them. E.g. ``str(session_id)``

        Parameters
        ----------
        saver_id : str | None, optional
            ID if multiple checkpoint savers are used

        Returns
        -------
        BaseCheckpointSaver
            Checkpoint saver instance
        """
        pass

    async def aget_checkpoint_saver(
        self, saver_id: str | None = None
    ) -> BaseCheckpointSaver:
        """Returns the requested checkpoint saver

        This is the asynchron version of :meth:`get_checkpoint_saver`.
        """
        return await asyncio.to_thread(
            self.get_checkpoint_saver,
            saver_id=saver_id,
        )
