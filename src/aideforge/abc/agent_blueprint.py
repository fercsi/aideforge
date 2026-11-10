from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aideforge.abc.agent_runtime import AbstractAgentRuntime
    from aideforge.abc.callback import AgentCallbackHandler
    from aideforge.abc.storage import AbstractStorage
    from aideforge.structs.agent import AgentConfig
    from aideforge.structs.llm import LlmConfig
    from aideforge.structs.session import AgentState
    from aideforge.utils import IdType
    from aideforge.tools import ToolBlueprintType

__all__ = [
    "AbstractAgentBlueprint",
]


class AbstractAgentBlueprint(ABC):
    """Blueprint for creating and configuring an AI agent.

    Defines the agent's identity, model, tools, system prompt, context
    configuration, and resources. Instances are templates used to create
    :class:`AgentRuntime` objects via the :meth:`instantiate` method.

    Parameters
    ----------
    config : AgentConfig, dict[str, Any] or None, optional
        The agent configuration. May be an :class:`AgentConfig`
        instance, a dictionary to validate, or ``None`` to use defaults.
    tool_pool : list[ToolBlueprintType] or None, optional
        A list of tools (tool blueprints) available to the agent. For the
        avaliable tool blueprint formats see :tool:`ToolBlueprintType`.
        If ``None``, no tools are available.
    storage : AbstractStorage or None, optional
        Storage backend for retrieving content resources. If ``None``,
        resources can be implemented only with their inline content.

    Attributes
    ----------
    name : str
        The agent's name, used for identification and lookup.
    config : AgentConfig
        The agent configuration.
    """

    def __init__(
        self,
        config: AgentConfig | dict[str, Any] | None = None,
        *,
        tool_pool: list[ToolBlueprintType] | None = None,
        storage: AbstractStorage | None = None,
    ) -> None:
        """Initialize an :class:`AbstractAgentBlueprint` instance.

        Parameters
        ----------
        config : AgentConfig, dict[str, Any] or None, optional
            The agent configuration. May be an :class:`AgentConfig`
            instance, a dictionary to validate, or ``None`` to use defaults.
        tool_pool : list[ToolBlueprintType] or None, optional
            A list of tools (tool blueprints) available to the agent. For the
            avaliable tool blueprint formats see :tool:`ToolBlueprintType`.
            If ``None``, no tools are available.
        storage : AbstractStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.

        Raises
        ------
        ValueError
            If *config* is provided but is neither an :class:`AgentConfig`
            nor a ``dict``.
        """
        self._config = config
        self._tool_pool = tool_pool
        self._storage = storage

    @property
    def name(self) -> str:
        """Agent type name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent's configuration."""
        return self._config

    @abstractmethod
    def instantiate(
        self,
        *,
        llm: LlmConfig | None = None,
        force_llm: bool = False,
    ) -> AbstractAgentRuntime:
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        llm : LlmConfig or None, optional
            If no model is configured in the agent blueprint, it must be
            specified here. By default, the agent's own model is used.
            If ``force_llm`` is ``True``, this parameter overrides the
            agent's default model.
        force_llm : bool, optional
            If ``True``, the provided *llm* overrides the agent's default
            model, by default ``False``.

        Returns
        -------
        AbstractAgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        TypeError
            If *llm* is ``None`` and ``force_llm`` is ``True`` (or the
            agent has no default model).
        ValueError
            If a tool is invalid, or if context configuration exceeds model
            limits.
        """
        ...

    @abstractmethod
    def restore(
        self,
        agent_id: IdType,
        state: AgentState,
    ) -> AbstractAgentRuntime:
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        agent_id : IdType
            Unique identifier for the agent.
        state : AgentState
            The agent state to restore.

        Returns
        -------
        AbstractAgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        ValueError
            If a tool is invalid, or if context configuration exceeds model
            limits.
        """
        ...
