from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aideforge.abc import AbstractStorage
    from aideforge.structs import (
        AgentConfig,
        AgentState,
        LlmConfig,
        Callbacks,
    )

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
    config : AgentConfig | dict[str, Any] | None, optional
        The agent configuration. May be an :class:`AgentConfig`
        instance, a dictionary to validate, or ``None`` to use defaults.
    tool_pool : list[ToolType] or None, optional
        A list of tools available to the agent. Each entry may be a
        :class:`BaseTool` subclass, a :class:`PreconfiguredTool`, or a
        callable. If ``None``, no tools are available.
    storage : AbstractStorage or None, optional
        Storage backend for retrieving content resources. If ``None``,
        resources can be implemented only with their inline content.

    Attributes
    ----------
    name : str
        The agent's name, used for identification and lookup.
    config : AgentConfig
        See above
    """

    def __init__(
        self,
        config: AgentConfig | dict[str, Any] | None = None,
        *,
        tool_pool: list[ToolType] | None = None,
        storage: AbstractStorage | None = None,
        callbacks: Callbacks | None = None,
    ) -> None:
        """Initialize an :class:`AgentBlueprint` instance.

        Creates copies of mutable class attributes so that each instance
        has its own independent tools and resources lists.

        Parameters
        ----------
        config : AgentConfig | dict[str, Any] | None, optional
            The agent configuration. May be an :class:`AgentConfig`
            instance, a dictionary to validate, or ``None`` to use defaults.
        tool_pool : list[ToolType] or None, optional
            A list of tools available to the agent. Each entry may be a
            :class:`BaseTool` subclass, a :class:`PreconfiguredTool`, or a
            callable. If ``None``, no tools are available.
        storage : AbstractStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.
        callbacks : Callbacks or None, optional
            One or more callback handlers to run during agent operation.

        Raises
        ------
        ValueError
            If *config* is provided but is neither an :class:`AgentConfig`
            nor a ``dict``.
        """
        self._config = config
        self._tool_pool = tool_pool
        self._storage = storage
        sekf._callbacks = callbacks

    @property
    def name(self) -> str:
        """Agent type name."""
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Agent's onfiguration."""
        return self._config

    @abstractmethod
    def instantiate(
        self,
        *,
        llm: LlmConfig | None = None,
        force_llm: bool = False,
    ) -> AgentRuntime:
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        llm : LlmConfig | None, optional
            If no model is configured in the agent blueprint, it must be
            specified here. By default, the agent's own model is used, but
            if ``force_llm`` is ``True``, this parameter overrides the
            agent's defaults.
        force_llm : bool, optional
            If ``True``, the provided *llm* overrides the agent's default
            model, by default ``False``.

        Returns
        -------
        AgentRuntime
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
        pass

    @abstractmethod
    def restore(
        self,
        agent_id: IdType,
        state: AgentState,
    ) -> AgentRuntime:
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        agent_id : IdType
        state : AgentState

        Returns
        -------
        AgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        ValueError
            If a tool is invalid, or if context configuration exceeds model
            limits.
        """
        pass

    @abstractmethod
    def update_callbacks(self, callbacks: Callbacks) -> None:
        pass
