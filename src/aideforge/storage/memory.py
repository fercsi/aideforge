from collections import defaultdict
from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from aideforge.structs import (
    BaseStruct,
    BaseIncrementalStruct,
    BaseEvent,
    ChangeProperty,
)
from .abc import BaseStorage, StructT

__all__ = ["MemoryStorage"]


class MemoryStorage(BaseStorage):
    checkpoint_saver: InMemorySaver = InMemorySaver()
    resources: dict[str, str | bytes] = {}
    structs: defaultdict[str, dict[str, BaseStruct]] = defaultdict(dict)

    def retrieve_resource(
        self,
        path: str,
        *,
        mode: str = "",
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
            Note: This parameter is not used in ``MemoryStorage``
        ignore_failure : bool, optional
            When ``True``, return ``None`` instead of raising if the
            resource cannot be found.

        Returns
        -------
        str or bytes or None
            The retrieved content, or ``None`` if not found
            and ``ignore_failure`` is ``True``.
        """
        if path not in MemoryStorage:
            if ignore_failure:
                return None
            else:
                raise FileNotFoundError(f"Resource '{path}' not found.")
        content = MemoryStorage.resources[path]
        return content

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
        MemoryStorage.resources[path] = content

    def retrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the filesystem and validate it.

        Parameters
        ----------
        struct_type : type[StructT]
            The BaseStruct class of the struct to retrieve.
        struct_name : str or None
            Name of the struct to retrieve.  When ``None``, it is derived
            from ``struct_type.storage_name``.
        ignore_failure : bool
            When ``True``, return ``None`` instead of raising if the struct
            cannot be found.

        Returns
        -------
        StructT or None
            The retrieved and validated struct, or ``None`` if not found
            and ``ignore_failure`` is ``True``.

        Raises
        ------
        FileNotFoundError
            If the struct is not found and ``ignore_failure`` is ``False``.
        """
        category, struct_name = _get_struct_keys(struct_type, struct_name)
        structs = MemoryStorage.structs
        if category in structs and struct_name in structs[category]:
            return structs[category][struct_name].model_copy(deep=True)
        else:
            if ignore_failure:
                return None
            else:
                raise FileNotFoundError(
                    f"'{struct_name}' of type {struct_type.__name__} not found."
                )

    def store_struct(
        self, struct: BaseStruct, struct_name: str | None = None
    ) -> None:
        """Store a struct on the filesystem.

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to store.
        struct_name : str or None
            Name under which the struct is stored.  When ``None``, it is
            derived from ``struct.storage_name``.
        """
        category, struct_name = _get_struct_keys(struct_type, struct_name)
        structs = MemoryStorage.structs
        structs[category][struct_name] = struct.model_copy(deep=True)

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
        category, struct_name = _get_struct_keys(struct_type, struct_name)
        structs = MemoryStorage.structs
        if category in structs and struct_name in structs[category]:
            # Apply to the stored one!
            struct = structs[category][struct_name]
        else:
            if ignore_failure:
                return None
            else:
                raise FileNotFoundError(
                    f"'{struct_name}' of type {struct_type.__name__} not found."
                )
        struct.events.append(event)
        if event.type == "change-property":
            chprop = cast(ChangeProperty, event)
            chprop.change(struct)

    def list_structs(self, struct_type: type[BaseStruct]) -> list[str]:
        """List all stored struct names for a given struct type.

        Parameters
        ----------
        struct_type : type[BaseStruct]
            The BaseStruct class whose stored instances should be listed.

        Returns
        -------
        list[str]
            Sorted list of struct names (without file extensions) found
            across all configured search paths.
        """
        category = struct_type.kinds or struct_type.kind
        structs = MemoryStorage.structs
        if category not in structs:
            return []
        return list(sorted(structs[category].keys()))

    def get_checkpoint_saver(self, saver_id:str | None = None) -> BaseCheckpointSaver:
        _ = saver_id
        return FileSystemStorage.checkpoint_saver

def _get_struct_keys(
    struct_type: type[StructT],
    struct_name: str | None = None,
) -> tuple[str, str]:
    if struct_name is None:
        struct_name = struct_type.storage_name.format(
            struct=_Proxy(struct_type)
        )
    category = struct_type.kinds or struct_type.kind
    return category, struct_name
