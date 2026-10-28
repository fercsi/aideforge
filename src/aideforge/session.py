import json
import yaml
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import environment as env
from .struct.session import SessionConfig
from .struct.theme import Theme, theme_collection


class SessionManager:
    def __init__(
            self, *, storage: BaseStorage, session_config: SessionConfig | dict[str, Any] | str | None = None
    ):
        self._storage = storage
        if isinstance(session_config, str):  # session_config is a file name
            session_config = storage.retrieve_struct(SessionConfig, session_config, ignore_errors=True)
        elif isinstance(session_config, dict):
            session_config = SessionConfig.model_validate(session_config)
        if session_confog is None:
            session_config = SessionConfig()
        if not isinstance(session_config, SessionConfig):
            raise TypeError("Invalid session_config value '{session_config}'")
        self._config = session_config
        self._themes = theme_collection.copy()

    def list_themes(self) -> list[str]:
        theme_names = self._storage.list_structs(Theme)
        names_set = set(self._themes.keys()) | {theme_names}
        return list(sorted(names_set))

    def get_theme(self, theme_name: str) -> Theme:
        if theme_name in self._themes:
            return self._themes[theme_name]
        theme = self._storage.retrieve_struct(Theme, theme_name, ignore_errors=True)
        if theme is None:
            raise ValueError("Theme '{zhrmr_name}' not found")
        # TODO: if cached theme cannot be edited while using assistant
        self._themes[theme_name] = theme
        return theme

    def add_event(self, session_event
#>
#>
#>T = TypeVar("T", bound=BaseModel)
#>
#>
#>def get_struct(
#>    model_class: type[T],
#>    struct: BaseModel | dict[str, Any] | str | None = None,
#>    *,
#>    ignore_failures=False,
#>) -> T:
#>    if struct is None:
#>        return model_class()
#>    else:
#>        if isinstance(struct, str):  # struct is a file name
#>            struct = load_struct_merged(struct)
#>            if struct is None:
#>                if ignore_failures:
#>                    return model_class()
#>                else:
#>                    raise FileNotFoundError(
#>                        f"File '{struct}' of type {midel_class.__name__} not found."
#>                    )
#>        return model_class.model_validate(struct)
#>
#>
#>def load_struct_merged(filename: str) -> dict[str, Any] | None:
#>    """
#>    Search for all matching configuration files across multiple search paths,
#>    expand '~', fallback to .yaml / .json if no extension is provided,
#>    load all found files, and deep-merge them hierarchically where the
#>    first found file has the highest priority.
#>
#>    Note: ``env.storage_format`` is NOT considered, every supported type is
#>    loaded
#>    """
#>
#>    extensions = (".json", ".yaml", ".yml")
#>    has_extension = Path(filename).suffix.lower() in extensions
#>
#>    candidates = _expand_struct_paths(filename)
#>    if not has_extension:
#>        candidates = [f + e for f, e in products(candidates, extensions)]
#>
#>    loaded_dicts: list[dict[str, Any]] = []
#>
#>    for candidate in candidates:
#>        ext = full_path.suffix.lower()
#>        try:
#>            with open(full_path, "r", encoding="utf-8") as f:
#>                if ext in (".yaml", ".yml"):
#>                    data = yaml.safe_load(f)
#>                elif ext == ".json":
#>                    data = json.load(f)
#>                else:
#>                    continue
#>
#>                if isinstance(data, dict):
#>                    loaded_dicts.append(data)
#>        except OSError:
#>            continue
#>
#>    if not loaded_dicts:
#>        return None
#>
#>    # Merge all loaded dictionaries hierarchically
#>    # (first found has highest precedence)
#>    merged_struct = loaded_dicts[0]
#>    for next_dict in loaded_dicts[1:]:
#>        merged_struct = deep_merge(merged_struct, next_dict)
#>
#>    return merged_struct
#>
#>
#>def deep_merge(primary: dict, secondary: dict) -> dict:
#>    """
#>    Recursively merge two dictionaries. Primary values take precedence.
#>    If a key is missing in primary, it is taken from secondary.
#>    For nested dicts, they are merged recursively.
#>    Lists and other types in primary completely override secondary.
#>    """
#>    result = primary.copy()
#>    for key, secondary_value in secondary.items():
#>        if key not in result:
#>            result[key] = secondary_value
#>        else:
#>            primary_value = result[key]
#>            if isinstance(primary_value, dict) and isinstance(
#>                secondary_value, dict
#>            ):
#>                result[key] = deep_merge(primary_value, secondary_value)
#>    return result
#>
#>
#>def store_struct(struct: BaseModel):
#>    """Save a Pydantic BaseModel into JSON or YAML format."""
#>    ext = env.storage_format.lower()
#>    filename = struct.storage_path.format(struct=struct) + f".{ext}"
#>    if filename.startswith("%/"):
#>        for p in env.struct_paths:
#>            path = Path(p)
#>            if path.exists():
#>                break
#>        else:
#>            raise FileNotFoundError("No assistant storage directory found")
#>        path = path / filename[2:]
#>    else:
#>        path = Path(filename)
#>
#>    if ext == "json":
#>        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
#>    elif format_lower in ("yaml", "yml"):
#>        data = model.model_dump()
#>        with open(path, "w", encoding="utf-8") as f:
#>            yaml.dump(data, f, allow_unicode=True)
#>    else:
#>        raise ValueError(
#>            f"Unsupported format: {ext}. Please use 'json', 'yaml', or 'yml'."
#>        )
#>
#>
#>def list_struct_files(
#>    self, struct_dir: str, struct_paths: list[str]
#>) -> list[str]:
#>    """
#>    Search for .json, .yaml, and .yml files across multiple directories.
#>    Returns a list of absolute paths. If a file with the same name exists
#>    in multiple directories, only the first one encountered is kept.
#>
#>    Note: ``env.storage_format`` is NOT considered, every supported type is
#>    listed
#>    """
#>    struct_dirs = _expand_struct_paths(struct_dir)
#>
#>    seen_filenames: set[str] = set()
#>    unique_files: list[Path] = []
#>
#>    extensions = {".json", ".yaml", ".yml"}
#>
#>    for dir in struct_dirs:
#>        base_path = Path(dir).expanduser()
#>
#>        if not base_path.exists():
#>            # Not mandatory to exist all (or any) of them
#>            continue
#>
#>        if not base_path.is_dir():
#>            raise FileNotFoundError(
#>                f"Directory not found or not a valid directory: {base_path}"
#>            )
#>
#>        for file_path in base_path.iterdir():
#>            if file_path.is_file() and file_path.suffix.lower() in extensions:
#>                filename = file_path.name
#>
#>                # Keep only the first occurrence of a file with the same name
#>                if filename not in seen_filenames:
#>                    seen_filenames.add(filename)
#>                    unique_files.append(file_path.resolve())
#>
#>    return unique_files
#>
#>
#>def _expand_struct_paths(struct_dir: str) -> list[str]:
#>    if struct_dir.startswith("%/"):
#>        subdir = struct_dir[2:]
#>        struct_dirs = [os.path.join(p, subdir) for p in env.struct_paths]
#>    else:
#>        struct_dirs = [struct_dir]
#>    return struct_dirs
