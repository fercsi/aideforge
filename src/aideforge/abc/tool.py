from typing import ClassVar, Protocol

__all__ = [
    "AgentToolProtocol",
]


class AgentToolProtocol(Protocol):
    """Protocol defining the interface for agentic tools in AideForge.

    Any tool used by an agent must implement this protocol to be discoverable
    and manageable by the agent runtime. The minimum requirements are a
    ``name`` and ``description`` class variable.

    .. note::
       Tools built on LangChain's ``BaseTool`` meet this requirement and can be
       used with AideForge.

    Attributes
    ----------
    name : str
        Unique identifier for the tool. Used for tool lookup, registration,
        and logging. Should be a valid Python identifier (e.g. ``"read_file"``,
        ``"web_search"``).
    description : str
        Human-readable description of what the tool does. Displayed to the
        agent/LLM to help it decide when to invoke the tool. Should be clear
        and concise (e.g. ``"Read the contents of a file from the filesystem"``,
        ``"Search the web for information"``).
    """

    name: ClassVar[str]
    description: ClassVar[str]
