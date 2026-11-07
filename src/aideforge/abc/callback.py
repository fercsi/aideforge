from abc import ABC, abstractmethod
from typing import Callable

class AgentCallbackHandler(ABC):
    #: Callback to use when some debug info should be sent to the interface
    on_debug: Callable[[str], None] | None = None
    #: Callback to use when some info should be sent to the interface
    on_info: Callable[[str], None] | None = None
    #: Callback to use when a warning should be sent to the interface
    on_warning: Callable[[str], None] | None = None
