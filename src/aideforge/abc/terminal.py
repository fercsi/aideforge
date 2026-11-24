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
    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...

#>    @abstractmethod
#>    def define_pane(self, name: str, region: Region) -> None:
#>        ...

    @abstractmethod
    def set_history(self, history: list[str]) -> None:
        ...

    @abstractmethod
    def set_theme(self, theme: Theme) -> None:
        ...

    @abstractmethod
    def input(
        self,
        prompt: str,
        *,
        style: str | None = None,
#>        format: TextFormat | None = None,
#>        pane: str = "main",
        **kwargs: Any,
    ) -> str:
        ...

    async def ainput(
        self,
        prompt: str,
        *,
        style: str | None = None,
#>        format: TextFormat | None = None,
#>        pane: str = "main",
        **kwargs: Any,
    ) -> str:
        return await asyncio.to_thread(
            self.input,
            prompt=prompt,
            style=style,
#>            format=format,
#>            pane=pane,
        )

    def print(
        self,
        *args: Any,
        smart_end: str | None = None,
#>        max_lines: int | None = None,
#>        indent: int = 0,
        style: str | None = None,
#>        format: TextFormat | None = None,
#>        pane: str = "main",
        **kwargs: Any,
    ) -> None:
        ...

    async def aprint(
        self,
        *args: Any,
        smart_end: str | None = None,
#>        max_lines: int | None = None,
#>        indent: int = 0,
        style: str | None = None,
#>        format: TextFormat | None = None,
#>        pane: str = "main",
        **kwargs: Any,
    ) -> None:
        return await asyncio.to_thread(
            self.print,
            *args,
            smart_end=smart_end,
#>            max_lines=max_lines,
#>            indent=indent,
            style=style,
#>            format=format,
#>            pane=pane,
            **kwargs,
        )
