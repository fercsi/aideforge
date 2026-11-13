import importlib
import inspect
from pathlib import Path

from aideforge.abc.command import BaseCommand

__all__ = [
    "default_commands",
]


def _import_commands() -> list[type[BaseCommand]]:
    """Dynamically import and discover :class:`BaseCommand` subclasses.

    Scans the current package directory for ``*.py`` files (excluding
    ``__init__.py``), imports each module, and collects all classes that
    are subclasses of :class:`BaseCommand` defined in that module.

    Returns
    -------
    list[type[BaseCommand]]
        A list of discovered command classes.
    """
    commands = []

    current_file = Path(__file__).resolve()
    package_dir = current_file.parent
    package_name = __package__ if __package__ else package_dir.name

    for file_path in package_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        # In case of some analytics tools, no package_name is present
        module_name = (
            f"{package_name}.{file_path.stem}"
            if package_name
            else file_path.stem
        )

        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseCommand)
                and obj is not BaseCommand
                and obj.__module__ == module.__name__
            ):
                commands.append(obj)
    return commands


default_commands = _import_commands()
