from abc import ABC, abstractmethod


class Lifecycle(ABC):
    """Lifecycle management interface

    This interface ensures that classes implementing it have synchron and
    asynchron lifecycle management feature including preparement and
    cleaning up operations.

    NOTE: Every function has both a synchronous and an asynchronous version.
    By default, each asynchronous version runs the corresponding synchronous
    function in a separate thread using ``asyncio.to_thread()``. In many
    cases, this is sufficient, and there is no need to implement a dedicated
    asynchronous version.
    """

    @abstractmethod
    def prepare(self) -> None:
        """Prepare operation"""
        ...

    async def aprepare(self) -> None:
        """Prepare operation

        This is the asynchron version of :meth:`prepare`.
        """
        await asyncio.to_thread(self.prepare)

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up operation"""
        ...

    async def acleanup(self) -> None:
        """Clean up operation

        This is the asynchron version of :meth:`cleanup`.
        """
        await asyncio.to_thread(self.cleanup)
