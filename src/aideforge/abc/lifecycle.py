import asyncio
from abc import ABC, abstractmethod

__all__ = [
    "Lifecycle",
]


class Lifecycle(ABC):
    """Lifecycle management interface.

    This interface ensures that classes implementing it have synchronous and
    asynchronous lifecycle management features including preparation and
    cleaning up operations.

    .. note::
       Every function has both a synchronous and an asynchronous version.
       By default, each asynchronous version runs the corresponding synchronous
       function in a separate thread using ``asyncio.to_thread()``. In many
       cases, this is sufficient, and there is no need to implement a dedicated
       asynchronous version.
    """

    def prepare(self) -> None:
        """Prepare operation."""
        ...

    async def aprepare(self) -> None:
        """Prepare operation.

        This is the asynchronous version of :meth:`prepare`.
        """
        await asyncio.to_thread(self.prepare)

    def cleanup(self) -> None:
        """Clean up operation."""
        ...

    async def acleanup(self) -> None:
        """Clean up operation.

        This is the asynchronous version of :meth:`cleanup`.
        """
        await asyncio.to_thread(self.cleanup)
