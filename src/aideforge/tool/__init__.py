import asyncio
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.tools import BaseTool

__all__ = ["AsyncTool", "ConfigurableTool"]


@dataclass
class ConfigurableTool:
    """A wrapper around a tool with associated configuration.

    Allows tools to be defined declaratively with a separate configuration
    dictionary that is applied during instantiation.

    Attributes
    ----------
    tool : BaseTool  Callable or custom classes
        The tool class or callable to be configured and instantiated.
    config : dict[str, Any] or None, optional
        Configuration parameters to pass to the tool during instantiation,
        or ``None`` if no configuration is needed.
    """

    tool: type | Callable
    config: dict[str, Any] | None = None


class AsyncTool(BaseTool):
    """Base class for creating asynchronous LangChain tools.

    Subclasses should implement the :meth:`_arun` method. The :meth:`_run`
    method bridges synchronous environments by running the async
    implementation, either by attaching to an existing running event loop
    (via :func:`asyncio.run_coroutine_threadsafe`) or by creating a new
    one with :func:`asyncio.run`.

    See Also
    --------
    langchain.tools.BaseTool : The LangChain base tool class.
    """

    # Base tool for asynchron tools. _run must be always implemented, so, this
    # is the default
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool synchronously by running its async counterpart.

        If a running asyncio event loop is detected in the current thread,
        the async task is scheduled on that loop and the result is awaited.
        Otherwise, :func:`asyncio.run` is used to execute ``_arun`` in a
        fresh event loop. It waits for the result and returns that.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :meth:`_arun`.
        **kwargs : Any
            Keyword arguments forwarded to :meth:`_arun`.

        Returns
        -------
        Any
            The result of the asynchronous tool execution.
        """
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


def _import_tools() -> list[type[BaseTool]]:
    """Dynamically import and discover :class:`BaseTool` subclasses.

    Scans the current package directory for ``*.py`` files (excluding
    ``__init__.py``), imports each module, and collects all classes that
    are subclasses of :class:`BaseTool` defined in that module.

    Returns
    -------
    list[type[BaseTool]]
        A list of discovered tool classes.
    """
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
