# TODO: expanduser wherever applicable. Maybe only in struct_paths?
# TODO: full_struct + .model_dumpi[_json](exclude_defaults=True)
import json
import os
import yaml
from itertools import product
from pathlib import Path
from typing import Any

from aideforge.structs.abc import BaseStruct
from aideforge.utils import deep_merge, remove_nones
from .abc import BaseStorage, StructT

__all__ = ["FileSystemStorage"]

EXTENSIONS = (".json", ".yaml", ".yml")


class FileSystemStorage(BaseStorage):
    """Storage backend that persists structs to the filesystem as JSON or YAML.

    Structs are written to and read from one or more configurable search
    directories (``struct_paths``).  When retrieving a struct, every matching
    file found across the search paths is deep-merged, with the first file
    encountered having the highest precedence.
    """

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

    def retrieve_struct(
        self,
        struct_type: type[StructT],
        struct_name: str | None = None,
        *,
        ignore_failures: bool = False,
    ) -> StructT | None:
        """Retrieve a struct from the filesystem and validate it.

        Parameters
        ----------
        struct_type : type[StructT]
            The BaseStruct class of the struct to retrieve.
        struct_name : str or None
            Name of the struct to retrieve.  When ``None``, it is derived
            from ``struct_type.storage_name``.
        ignore_failures : bool
            When ``True``, return ``None`` instead of raising if the struct
            cannot be found.

        Returns
        -------
        StructT or None
            The retrieved and validated struct, or ``None`` if not found
            and ``ignore_failures`` is ``True``.

        Raises
        ------
        FileNotFoundError
            If the struct is not found and ``ignore_failures`` is ``False``.
        """
        if struct_name is None:
            struct_name = struct_type.storage_name.format(
                struct=_Proxy(struct_type)
            )
        if struct_type.kinds is not None:
            filename = f"%/{struct_type.kinds}/{struct_name}"
        else:
            filename = f"%/{struct_name}"
        struct = self._load_struct_merged(filename)
        if struct is None:
            if ignore_failures:
                return None
            else:
                raise FileNotFoundError(
                    f"'{struct_name}' of type {struct_type.__name__} not found."
                )
        return struct_type.model_validate(struct)

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
        if struct_name is None:
            struct_name = struct.storage_name.format(struct=struct)
        ext = self.struct_format.lower()
        if struct.kinds is not None:
            filename = f"%/{struct.kinds}/{struct_name}.{ext}"
        else:
            filename = f"%/{struct_name}.{ext}"
        self._save_struct(struct, filename)

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
        if struct_type.kinds is not None:
            struct_dir = f"%/{struct_type.kinds}"
            files = self._list_struct_files(struct_dir)
        else:
            files = self._search_struct_path_alternatives(
                f"%/{struct_type.storage_name}"
            )
        return list(sorted({Path(f).stem for f in files}))

    # private methods

    def _load_struct_merged(self, filename: str) -> dict[str, Any] | None:
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
                        data = yaml.safe_load(f)
                    elif ext == ".json":
                        data = json.load(f)
                    else:
                        continue

                    if isinstance(data, dict):
                        loaded_dicts.append(data)
            except OSError:
                continue

        if not loaded_dicts:
            return None

        # Merge all loaded dictionaries hierarchically
        # (first found has highest precedence)
        merged_struct = loaded_dicts[0]
        for next_dict in loaded_dicts[1:]:
            merged_struct = deep_merge(merged_struct, next_dict)

        return merged_struct

    def _save_struct(
        self, struct: BaseStruct, filename: str, *, full_struct: bool = False
    ) -> None:
        """Save a Pydantic BaseStruct into JSON or YAML format.

        Parameters
        ----------
        struct : BaseStruct
            The BaseStruct instance to save.
        filename : str
            Relative filename prefixed with ``"%/"`` or an absolute path
            where the struct should be written.
        """
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

        path.parent.mkdir(parents=True, exist_ok=True)
        ext = self.struct_format.lower()
        if ext == "json":
            json_str = struct.model_dump_json(
                exclude_none=not full_struct, indent=2
            )
            path.write_text(
                json_str,
                encoding="utf-8",
            )
        elif ext in ("yaml", "yml"):
            data = struct.model_dump()
            if not full_struct:
                data = remove_nones(data)
            with open(path, "w", encoding="utf-8") as f:
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


class _Proxy:
    """Internal proxy that makes a ``BaseStruct`` subclass behave like an
    instance of its default values."""

    def __init__(self, target: type[BaseStruct]) -> None:
        """Initialize the proxy for a given struct class.

        Parameters
        ----------
        target : type[BaseStruct]
            The ``BaseStruct`` subclass to proxy.
        """
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
            raise AttributeError(f"'{self._target.__name__}' object has no attribute '{name}'")
        return field_info.default
