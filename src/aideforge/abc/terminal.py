import asyncio
from typing import Any, TypeVar, abstractmethod

from aideforge.structs.theme import Theme
from .lifecycle import Lifecycle

__all__ = ["Terminal"]


#>class Region(BaseModel):
#>    x1: int
#>    y1: int
#>    x2: int
#>    y2: int

class Terminal(Lifecycle):
    """Abstract base class for terminal implementations.

    Provides an interface for terminal input/output operations with
    theme support and history management.
    """

    @property
    @abstractmethod
    def width(self) -> int:
        """Width of the terminal in characters.

        Returns
        -------
        int
            Terminal width.
        """
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Height of the terminal in characters.

        Returns
        -------
        int
            Terminal height.
        """
        ...

#>    @abstractmethod
#>    def define_pane(self, name: str, region: Region) -> None:
#>        ...

    @abstractmethod
    def set_history(self, history: list[str]) -> None:
        """Set the command history.

        Parameters
        ----------
        history : list[str]
            List of history entries to load.
        """
        ...

    @abstractmethod
    def set_theme(self, theme: Theme) -> None:
        """Apply a color theme to the terminal.

        Parameters
        ----------
        theme : Theme
            Theme configuration to apply.
        """
        ...

    @abstractmethod
    def input(
        self,
        prompt: str,
        *,
        style: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Read a line of input from the user.

        Parameters
        ----------
        prompt : str
            Prompt string to display.
        style : str or None, optional
            Style name to apply to the prompt, by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.

        Returns
        -------
        str
            The user's input line.
        """
        ...

    async def ainput(
        self,
        prompt: str,
        *,
        style: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously read a line of input from the user.

        Parameters
        ----------
        prompt : str
            Prompt string to display.
        style : str or None, optional
            Style name to apply to the prompt, by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.

        Returns
        -------
        str
            The user's input line.
        """
        return await asyncio.to_thread(
            self.input,
            prompt=prompt,
            style=style,
        )

    def print(
        self,
        *args: Any,
        smart_end: str | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Print output to the terminal.

        Parameters
        ----------
        *args : Any
            Objects to print.
        smart_end : str or None, optional
            Ending string if the last argument doesn't already end with it,
            by default None.
        style : str or None, optional
            Style name to apply to the output, by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.
        """
        ...

    async def aprint(
        self,
        *args: Any,
        smart_end: str | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Asynchronously print output to the terminal.

        Parameters
        ----------
        *args : Any
            Objects to print.
        smart_end : str or None, optional
            Ending string if the last argument doesn't already end with it,
            by default None.
        style : str or None, optional
            Style name to apply to the output, by default None.
        **kwargs : Any
            Additional arguments passed to the underlying implementation.
        """
        return await asyncio.to_thread(
            self.print,
            *args,
            smart_end=smart_end,
            style=style,
            **kwargs,
        )