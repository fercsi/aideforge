__all__ = ["AsyncTool", "ConfigurableTool"]

import asyncio
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from langchain.tools import BaseTool
from pathlib import Path
from typing import Any


@dataclass
class ConfigurableTool:
    tool: type | Callable
    config: dict[str, Any] | None = None


class AsyncTool(BaseTool):
    # Base tool for asynchron tools. _run must be always implemented, so, this
    # is the default
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # Check if thread has a running asyncio (a single instance is enabéed)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Pass task to running instance
            future = asyncio.run_coroutine_threadsafe(
                self._arun(*args, **kwargs), loop
            )
            # and wait for result (no await is possible here)
            return future.result()

        # if no running asyncio
        return asyncio.run(self._arun(*args, **kwargs))


def _import_tools() -> list[BaseTool]:
    tools = []

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
                issubclass(obj, BaseTool)
                and obj is not BaseTool
                and obj.__module__ == module.__name__
            ):
                tools.append(obj)
    return tools


default_tools = _import_tools()
