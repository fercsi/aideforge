from typing import Any, TypeVar, overload


def deep_merge(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Primary values take precedence.  If a key is missing in *primary*, it
    is taken from *secondary*.  For nested dicts, they are merged
    recursively.  Lists and other types in *primary* completely override
    *secondary*.

    Parameters
    ----------
    primary : dict
        The primary dictionary whose values take precedence during the
        merge.
    secondary : dict
        The secondary dictionary whose values are used only when the
        corresponding key is absent in *primary*.

    Returns
    -------
    dict
        A new dictionary containing the deep-merged result of *primary*
        and *secondary*.
    """
    result = primary.copy()
    for key, secondary_value in secondary.items():
        if key not in result:
            result[key] = secondary_value
        else:
            primary_value = result[key]
            if isinstance(primary_value, dict) and isinstance(
                secondary_value, dict
            ):
                result[key] = deep_merge(primary_value, secondary_value)
    return result


T = TypeVar("T")


@overload
def remove_nones(data: dict[Any, Any]) -> dict[Any, Any]:
    ...


@overload
def remove_nones(data: list[Any]) -> list[Any]:
    ...


@overload
def remove_nones(data: T) -> T:
    ...


def remove_nones(data: Any) -> Any:
    """Recursively remove ``None`` values from a nested data structure.

    Traverses dictionaries and lists, omitting any entries whose value
    is ``None``.  Non-container values are returned unchanged.

    Parameters
    ----------
    data : Any
        The data structure to clean.  May be a ``dict``, ``list``, or any
        other type.

    Returns
    -------
    Any
        A new data structure of the same type as *data* with all ``None``
        values removed.
    """
    if isinstance(data, dict):
        return {k: remove_nones(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_nones(v) for v in data if v is not None]
    return data
