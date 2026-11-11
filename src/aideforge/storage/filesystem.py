# TODO: expanduser wherever applicable. Maybe only in struct_paths?
# TODO: figure out which EXCLUSIONS to use: unset, defaults, none, computed_fields

import json
import os
import yaml
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

from aideforge.abc.storage import (
    AbstractStorage,
    AgentCheckpointStoreProtocol,
    StructT,
)
from aideforge.structs.abc import (
    BaseStruct,
    BaseIncrementalStruct,
    BaseEvent,
    ChangeProperty,
    StructProxy,
)
from aideforge.utils import deep_merge, remove_nones

try:
    import aiosqlite
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    LANGGRAPH_SUPPORTED = True
except:
    LANGGRAPH_SUPPORTED = False

__all__ = ["FileSystemStorage"]

EXTENSIONS = (".json", ".jsonl", ".yaml", ".yml")
EXCLUSIONS = {}

class DatabaseConnection(Protocol):
    """ Database connection semantif type marker

    The protocol serves as a semantic type marker, identifying objects
    intended to provide database connection without imposing any
    implementation requirements. Concrete integrations are responsible for
    ensuring compatibility with the agent runtime that consumes them.
    """
    ...


class FileSystemStorage(AbstractStorage):
    """Storage backend that persists structs to the filesystem as JSON or YAML.

    Structs are written to and read from one or more configurable search
    directories (``struct_paths``).  When retrieving a struct, every matching
    file found across the search paths is deep-merged, with the first file
    encountered having the highest precedence.
    """

    _checkpoint_store: ClassVar[AgentCheckpointStoreProtocol | None] = None
    _conn: ClassVar[DatabaseConnection | None] = None

    def __init__(
        self,
        *,
        struct_paths: list[str] | None = None,
        struct_format: str = "json",
    ) -> None:
        """Initialize the filesystem storage backend.

        Parameters
        ----------
        struct_paths : list[str] or None
            List of directories used as search paths for storing and loading
            structs.  When ``None``, an empty list is used (no paths
            configured).
            E.g. ``["./aideforge", "~/.aideforge", "/etc/aideforge"]``
        struct_format : str
            File format for storing structs.  Supported values are
            ``"json"``, ``"yaml"`` and ``"yml"``.
        """
        # Beyond constructor No runtime validation (internally used method;
        # assumes valid inputs)
        self.struct_paths = [] if struct_paths is None else struct_paths.copy()
        self.struct_format = struct_format
        FileSystemStorage._conn = None

    def __del__(self) -> None:
        # TODO: what should we do in async case if acleanup has not been called
        if FileSystemStorage._conn:
            FileSystemStorage._conn.close()

    def prepare(self) -> None:
        if LANGGRAPH_SUPPORTED:
            dbpath = self._get_path(None, "checkpoint.db")
            FileSystemStorage._conn = sqlite3.connect(dbpath)
            FileSystemStorage._checkpoint_store = SqliteSaver(FileSystemStorage._conn)

    async def aprepare(self) -> None:
        if LANGGRAPH_SUPPORTED:
            dbpath = self._get_path(None, "checkpoint.db")
            FileSystemStorage._conn = await aiosqlite.connect(dbpath)
            FileSystemStorage._checkpoint_store = AsyncSqliteSaver(FileSystemStorage._conn)

    def cleanup(self) -> None:
        if FileSystemStorage._conn:
            FileSystemStorage._conn.close()
        FileSystemStorage._conn = None

    async def acleanup(self) -> None:
        if FileSystemStorage._conn:
            await FileSystemStorage._conn.close()
        FileSystemStorage._conn = None

    def retrieve_resource(
        self,
        path: str,
        *,
        mode: str = "t",
        ignore_failure: bool = False,
    ) -> str | bytes | None:
        if mode not in ("b", "t"):
            raise ValueError(f"Invalid retrieve mode '{mode}'")
        if path[0].isalpha():
            path = "%/" + path
        alternatives = self._search_struct_path_alternatives(self, path)
        if not alternatives:
            if ignore_failure:
                return None
            else:
                raise FileNotFoundError(f"Resource '{path}' not found.")
        try:
            with open(file, f"r{mode}") as f:
                content = f.read()
            return content
        except OSError:
            if ignore_failure:
                return None
            else:
                raise

    def store_resource(
        self,
        path: str,
        content: str | bytes,
        *,
        ignore_failure: bool = False,
    ) -> None:
        path = self._get_path(struct, struct_name)
        mode = "wt" if isinstance(content, str) else "wb"
        try:
            with open(file, mode) as f:
                f.write(content)
        except OSError:
            if ignore_failure:
                return None
            else:
                raise

    def retrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failure: bool = False,
    ) -> StructT | None:
        if struct_name is None:
            struct_name = struct_type.storage_name.format(
                struct=StructProxy(struct_type)
            )
        if struct_type.kinds is not None:
            filename = f"%/{struct_type.kinds}/{struct_name}"
        else:
            filename = f"%/{struct_name}"
        data = self._load_struct_merged(filename)
        if data is None:
            if ignore_failure:
                return None
            else:
                raise FileNotFoundError(
                    f"'{struct_name}' of type {struct_type.__name__} not found."
                )
        if isinstance(data, list):  # incremental data
            struct = struct_type.model_validate(data[0])
            events = [BaseEvent.model_validate(e) for e in data[1:]]
            self._resolve_incremental(struct, events)
        else:
            struct = struct_type.model_validate(data)
        return struct

    def store_struct(
        self, struct: BaseStruct, struct_name: str | None = None
    ) -> None:
        path = self._get_path(struct, struct_name)
        self._save_struct(struct, path)

    def store_struct_increment(
        self,
        struct: BaseIncrementalStruct,
        increment: BaseEvent,
        *,
        struct_name: str | None = None,
    ) -> None:
        path = self._get_path(struct, struct_name)
        self._append_struct_increment(struct, increment, path)

    def list_structs(self, struct_type: type[BaseStruct]) -> list[str]:
        if struct_type.kinds is not None:
            struct_dir = f"%/{struct_type.kinds}"
            files = self._list_struct_files(struct_dir)
        else:
            files = self._search_struct_path_alternatives(
                f"%/{struct_type.storage_name}"
            )
        return list(sorted({Path(f).stem for f in files}))

    def get__checkpoint_store(
        self, store_id: str | None = None
    ) -> AgentCheckpointStoreProtocol:
        _ = store_id
        if FileSystemStorage._checkpoint_store is None:
            self.prepare()
        return FileSystemStorage._checkpoint_store

    async def aget__checkpoint_store(
        self, store_id: str | None = None
    ) -> AgentCheckpointStoreProtocol:
        _ = store_id
        if FileSystemStorage._checkpoint_store is None:
            await self.aprepare()
        return FileSystemStorage._checkpoint_store

    # private methods

    def _load_struct_merged(
        self, filename: str
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Search for all matching configuration files across multiple search paths,
        expand '~', fallback to .yaml / .json if no extension is provided,
        load all found files, and deep-merge them hierarchically where the
        first found file has the highest priority.

        Parameters
        ----------
        filename : str
            Relative filename prefixed with ``"%/"`` to indicate it should
            be searched across all configured struct paths, or an absolute
            path.

        Returns
        -------
        dict[str, Any] or None
            The deep-merged dictionary loaded from all matching files, or
            ``None`` if no files were found.

        Notes
        -----
        ``self.struct_format`` is NOT considered; every supported extension
        (``.json``, ``.yaml``, ``.yml``) is loaded.
        """
        files = self._search_struct_path_alternatives(filename)
        loaded_dicts = []
        for file in files:
            ext = Path(file).suffix.lower()
            try:
                with open(file, "r", encoding="utf-8") as f:
                    if ext in (".yaml", ".yml"):
                        data = list(yaml.safe_load_all(f))
                        data = data[0] if len(data) == 1 else data
                    elif ext == ".json":
                        data = json.load(f)
                    elif ext == ".jsonl":
                        data = [json.loads(l) for l in f]
                        data = data[0] if len(data) == 1 else data
                    else:
                        continue

                    if isinstance(data, dict):
                        loaded_dicts.append(data)
                    elif isinstance(data, list):
                        loaded_dicts.append(data)
                        # Incremental storage does not support merging now.
                        # return first struct found as is.
                        return loaded_dicts[0]
                    # If a file contains invalid data, we just skip it
            except OSError:
                # Don't deal with OS errors, most candidates do not even exist
                continue

        if not loaded_dicts:
            return None

        # Merge all loaded dictionaries hierarchically
        # (first found has highest precedence)
        merged_struct = loaded_dicts[0]
        for next_dict in loaded_dicts[1:]:
            merged_struct = deep_merge(merged_struct, next_dict)

        return merged_struct

    def _resolve_incremental(
        self, struct: BaseIncrementalStruct, events: list[BaseEvent]
    ) -> None:
        for event in events:
            struct.events.append(event)
            if event.type != "change-property":
                continue
            chprop = cast(ChangeProperty, event)
            chprop.change(struct)

    def _get_path(
        self, struct: BaseStruct | None, struct_name: str | None
    ) -> Path:
        if struct_name is None:
            struct_name = struct.storage_name.format(struct=struct)
        ext = self.struct_format.lower()
        if struct and struct.storage_mode == "incremental" and ext == "json":
            ext = "jsonl"
        if struct is None:
            filename = f"%/{struct_name}"
        elif struct.kinds is not None:
            filename = f"%/{struct.kinds}/{struct_name}.{ext}"
        else:
            filename = f"%/{struct_name}.{ext}"

        if filename.startswith("%/"):
            for p in self.struct_paths:
                path = Path(p)
                if path.exists():
                    break
            else:
                # create last directory in list
                # TODO: if /etc is the last??
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError:
                    raise FileNotFoundError(
                        "No assistant storage directory found"
                    )
            path = path / filename[2:]
        else:
            path = Path(filename)
        return path

    def _save_struct(self, struct: BaseStruct, path: Path) -> None:
        """Save a Pydantic BaseStruct into JSON or YAML format.

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to save.
        path : Path
            Path where the struct should be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = self.struct_format.lower()
        if ext == "json":
            indent = None if struct.storage_mode == "incremental" else 2
            json_str = struct.model_dump_json(indent=indent, **EXCLUSIONS)
            path.write_text(
                json_str,
                encoding="utf-8",
                newline="\n",
            )
        elif ext in ("yaml", "yml"):
            data = struct.model_dump(**EXCLUSIONS)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            raise ValueError(
                f"Unsupported format: {ext}. Please use 'json', 'yaml', or 'yml'."
            )

    def _append_struct_increment(
        self, struct: BaseStruct, increment: BaseStruct, path: Path
    ) -> None:
        """Save a Pydantic BaseStruct into JSON or YAML format.

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to save.
        path : Path
            Path where the struct should be written.
        """
        ext = self.struct_format.lower()
        if ext == "json":
            json_str = increment.model_dump_json(**EXCLUSIONS)
            with path.open("a", encoding="utf-8") as f:
                f.write(json_str + "\n")
        elif ext in ("yaml", "yml"):
            data = struct.model_dump(**EXCLUSIONS)
            with path.open("a", encoding="utf-8") as f:
                f.write("---\n")
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            raise ValueError(
                f"Unsupported format: {ext}. Please use 'json', 'yaml', or 'yml'."
            )

    def _list_struct_files(self, struct_dir: str) -> list[str]:
        """Search for .json, .yaml, and .yml files across multiple directories.
        Returns a list of absolute paths. If a file with the same name exists
        in multiple directories, only the first one encountered is kept.

        Parameters
        ----------
        struct_dir : str
            Directory (with ``"%/"`` prefix or absolute path) to search for
            struct files.

        Returns
        -------
        list[str]
            List of absolute paths to matching ``.json``, ``.yaml``, and
            ``.yml`` files.  If a file with the same name exists in multiple
            directories, only the first one encountered is kept.

        Notes
        -----
        ``self.struct_format`` is NOT considered; every supported extension is
        listed.
        """
        struct_dirs = self._expand_struct_paths(struct_dir)

        seen_filenames: set[str] = set()
        unique_files: list[Path] = []

        for dir in struct_dirs:
            base_path = Path(dir).expanduser()

            if not base_path.exists():
                # Not mandatory to exist all (or any) of them
                continue

            if not base_path.is_dir():
                raise FileNotFoundError(
                    f"Directory not found or not a valid directory: {base_path}"
                )

            for file_path in base_path.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in EXTENSIONS
                ):
                    filename = file_path.name

                    # Keep only the first occurrence of a file with the same name
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        unique_files.append(file_path.resolve())

        return [str(p) for p in unique_files]

    def _search_struct_path_alternatives(self, filename: str) -> list[str]:
        """Expand a filename into candidate paths across all struct paths.

        Parameters
        ----------
        filename : str
            Filename to search for, optionally prefixed with ``"%/"``.

        Returns
        -------
        list[str]
            List of existing paths matching the filename, expanded across
            all struct paths and with all supported extensions appended if
            no extension was provided.
        """
        has_extension = Path(filename).suffix.lower() in EXTENSIONS

        candidates = self._expand_struct_paths(filename)
        if not has_extension:
            candidates = [f + e for f, e in product(candidates, EXTENSIONS)]

        paths_matched = []

        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                paths_matched.append(candidate)

        return paths_matched

    def _expand_struct_paths(self, struct_path: str) -> list[str]:
        """Expand a struct path into one or more concrete paths.

        Parameters
        ----------
        struct_path : str
            Path to expand.  When prefixed with ``"%/"``, the remainder is
            appended to every configured struct path.

        Returns
        -------
        list[str]
            List of resolved paths to search.
        """
        if struct_path.startswith("%/"):
            subdir = struct_path[2:]
            struct_paths = [os.path.join(p, subdir) for p in self.struct_paths]
        else:
            struct_paths = [struct_path]
        return struct_paths
