import asyncio
from abc import abstractmethod
from typing import Protocol, TypeVar

from aideforge.structs.abc import BaseStruct, BaseIncrementalStruct, BaseEvent
from .lifecycle import Lifecycle

__all__ = ["AbstractStorage", "AgentCheckpointStoreProtocol", "StructT"]

StructT = TypeVar("StructT", bound=BaseStruct)


class AgentCheckpointStoreProtocol(Protocol):
    """Protocol representing an agent checkpoint storage mechanism.

    This protocol intentionally does not define any methods or attributes.
    Different agent frameworks and runtimes may implement checkpointing
    using fundamentally different interfaces and concepts. Defining a common
    interface here would unnecessarily constrain integrations and introduce
    framework-specific assumptions.

    The protocol therefore serves as a semantic type marker, identifying
    objects intended to provide checkpoint storage without imposing any
    implementation requirements. Concrete integrations are responsible for
    ensuring compatibility with the agent runtime that consumes them.
    """
    ...


class AbstractStorage(Lifecycle):
    """Abstract base class for struct storage backends.

    A storage backend is responsible for persisting, retrieving, and listing
    struct instances (``BaseStruct`` subclasses) in a concrete medium (e.g.
    the filesystem, a database, or a remote API).

    .. note::
       Every function has both a synchronous and an asynchronous version.
       By default, each asynchronous version runs the corresponding
       synchronous function in a separate thread using
       ``asyncio.to_thread()``. In many cases, this is sufficient, and there
       is no need to implement a dedicated asynchronous version.
    """

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
        str, bytes or None
            The retrieved content, or ``None`` if not found
            and ``ignore_failure`` is ``True``.
        """
        ...

    async def aretrieve_resource(
        self,
        path: str,
        *,
        mode: str = "t",
        ignore_failure: bool = False,
    ) -> str | bytes | None:
        """Retrieve a resource from the storage backend.

        This is the asynchronous version of :meth:`retrieve_resource`.
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
            Path of the resource to store.
        content : str or bytes
            Content of the resource to be stored.
        ignore_failure : bool
            When ``True``, return instead of raising if the
            resource cannot be stored.
        """
        ...

    async def astore_resource(
        self,
        path: str,
        content: str | bytes,
        *,
        ignore_failure: bool = False,
    ) -> None:
        """Store a resource to the storage backend.

        This is the asynchronous version of :meth:`store_resource`.
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
        ...

    async def aretrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the storage backend.

        This is the asynchronous version of :meth:`retrieve_struct`.
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
        ignore_failure : bool
            When ``True``, return instead of raising if the
            struct cannot be stored.
        """
        ...

    async def astore_struct(
        self,
        struct: BaseStruct,
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> None:
        """Store a struct in the storage backend.

        This is the asynchronous version of :meth:`store_struct`.
        """
        return await asyncio.to_thread(
            self.store_struct,
            struct=struct,
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
        """Store an increment to an incrementally stored struct.

        Parameters
        ----------
        struct : BaseIncrementalStruct
            The BaseIncrementalStruct instance to store.
        increment : BaseEvent
            The BaseEvent increment instance to store within ``struct``'s
            scope.
        struct_name : str or None
            Name under which the struct is stored.  When ``None``, a
            default name is derived from ``struct.storage_name``.
        ignore_failure : bool
            When ``True``, return instead of raising if the
            increment cannot be stored.
        """
        ...

    async def astore_struct_increment(
        self,
        struct: BaseIncrementalStruct,
        increment: BaseEvent,
        *,
        struct_name: str | None = None,
        ignore_failure: bool = False,
    ) -> None:
        """Store an increment to an incrementally stored struct.

        This is the asynchronous version of :meth:`store_struct_increment`.
        """
        return await asyncio.to_thread(
            self.store_struct_increment,
            struct=struct,
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
            The BaseStruct class whose stored instances should be listed.

        Returns
        -------
        list[str]
            Sorted list of struct names found in the storage backend.
        """
        ...

    async def alist_structs(self, struct_type: type[BaseStruct]) -> list[str]:
        """List all stored struct names for a given struct type.

        This is the asynchronous version of :meth:`list_structs`.
        """
        return await asyncio.to_thread(
            self.list_structs,
            struct_type=struct_type,
        )

    @abstractmethod
    def get_checkpoint_store(
        self, store_id: str | None = None
    ) -> AgentCheckpointStoreProtocol | None:
        """Return the requested checkpoint store.

        Usually a single checkpoint store for the whole application is
        sufficient. If multiple stores are needed, use ``store_id`` to
        identify them. E.g. ``str(session_id)``.

        Parameters
        ----------
        store_id : str or None, optional
            ID if multiple checkpoint stores are used.

        Returns
        -------
        AgentCheckpointStoreProtocol
            Checkpoint store instance.
        """
        ...

    async def aget_checkpoint_store(
        self, store_id: str | None = None
    ) -> AgentCheckpointStoreProtocol | None:
        """Return the requested checkpoint store.

        This is the asynchronous version of :meth:`get_checkpoint_store`.
        """
        return await asyncio.to_thread(
            self.get_checkpoint_store,
            store_id=store_id,
        )
