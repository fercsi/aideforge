from abc import ABC, abstractmethod
from typing import Callable

__all__ = [
    "AgentCallbackHandler",
]


class AgentCallbackHandler(ABC):
    """Callback handler interface for agent events.

    Implement this class to receive callbacks from the agent during
    execution. All callbacks are optional and default to ``None``.

    Attributes
    ----------
    on_debug : Callable[[str], None] or None
        Called when debug information should be sent to the interface.
    on_info : Callable[[str], None] or None
        Called when information should be sent to the interface.
    on_warning : Callable[[str], None] or None
        Called when a warning should be sent to the interface.
    """

    on_debug: Callable[[str], None] | None = None
    on_info: Callable[[str], None] | None = None
    on_warning: Callable[[str], None] | None = None
