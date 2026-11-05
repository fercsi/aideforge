from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import AsyncGeneratot, Generator

    from aideforge.results import LlmError, LlmResponse
    from aideforge.storage import BaseStorage
    from aideforge.structs import (
        AgentConfig,
        AgentState,
        ActionRequest,
        ApprovalDecision,
        LlmConfig,
        Callbacks,
    )
    from aideforege.tools import ToolPolicy

__all__ = [
    "AbstractAgentRuntime",
    "AbstractAgentBlueprint",
]


class AbstractAgentRuntime(ABC):
    """Runtime representation of an instantiated agent.

    An ``AgentRuntime`` object is created by the :class:`AgentBlueprint`,
    which is a template that allows multiple identical agents to be created.
    The object stores all the information required for the agent's operation
    and provides an interface for communicating with the model
    (:meth:`process_prompt`).

    The input parameters are created by the agent blueprint in various forms
    and control the agent's behavior.

    Note: Never create an ``AgentRuntime`` object without using
    ``AgentBlueprint``.

    Warning: Never use an AgentRuntime instance in multiple sessions. Create
    one for each of them.

    Parameters
    ----------
    setup : AgentSetup
        The resolved agent setup containing model config, tools, system
        message, and callback configuration.
    config : AgentConfig
        The agent configuration template with tool policies, tools,
        system prompt, context, and LLM defaults.
    llm_config : LlmConfig
        Large language model's configuration data
    tool_policy : dict[str, ToolPolicy]
        Tool policy information for each tools
    threads : list[IdType]
        List of LangGraph thread IDs

    Properties
    ----------
    name : str
        Name of the agent's type
    agent_id : IdType
        Unique ID of agent
    state : AgentState
        State information about the agent. Note, that LangChain
    context_info : ContextInfo
        Information on context. Rather limited at this point (contains
        max_input_tokens and max_output_tokens)
    setup : AgentSetup
        See above
    config : AgentConfig
        See above
    """

    def __init__(
        self,
        *,
        setup: AgentSetup,
        config: AgentConfig,
        storage: BaseStorage | None = None,
    ) -> None:
        """Initialize the agent runtime with resolved configuration.

        Parameters
        ----------
        setup : AgentSetup
            The resolved agent setup containing model config, tools,
            system message, and callback configuration.
        config : AgentConfig
            The agent configuration containing tool policies, tools,
            system prompt, context, and LLM defaults.
        storage : BaseStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.
        """
        self._setup = setup
        self._config = config
        self._storage = storage
        self._agent_id = setup.agent_id
        pass

    @property
    def name(self) -> str:
        """Agent type name."""
        return self._config.name

    @property
    def agent_id(self) -> IdType:
        """Agent's unique ID."""
        return self._agent_id

    @property
    @abstractmethod
    def state(self) -> AgentState:
        """Agent's current state."""
        pass

    @abstractmethod
    def prompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> Generator[ActionRequest, ApprovalDecision, LlmResponse | LlmError]:
        """Process a user prompt through the LLM agent in synchron mode.

        Parameters
        ----------
        prompt : str
            The user's input prompt text.
        thread_idx : int, optional
            The index of the conversation thread to use, by default 0.
            If the thread does not exist yet, it is created automatically.

        Yields
        ------
        ActionRequest
            ``ActionRequest`` instances are yielded when the agent needs
            human approval to invoke a tool; the caller must send an
            :class:`ApprovalDecision` via ``asend``.

        Returns
        -------
        LlmResponse | LlmError
            ``LlmResponse`` is returned with the final LLM response content
            on success. ``LlmError`` is returned if an exception occurs during
            processing.

        Examples
        --------
        >>> processor = assistant.prompt_processor("List me the python files")
        >>> try:
        ...   action_req = next(processor)
        ...   while True:
        ...     action = input(action_req.description + "\napprove/reject? ")
        ...     action_req = processor.send(ApprovalDecision(action))
        ... except StopIteration as exc
        ...     response = exc.value
        >>> # process response LLM message, command output or error
        """
        pass

    @abstractmethod
    async def aprompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> AsyncGenerator[
        ActionRequest | LlmResponse | LlmError, ApprovalDecision
    ]:
        """Process a user prompt through the LLM agent in asynchron mode.

        Parameters
        ----------
        prompt : str
            The user's input prompt text.
        thread_idx : int, optional
            The index of the conversation thread to use, by default 0.
            If the thread does not exist yet, it is created automatically.

        Yields
        ------
        ActionRequest | LlmResponse | LlmError
            ``ActionRequest`` instances are yielded when the agent needs
            human approval to invoke a tool; the caller must send an
            :class:`ApprovalDecision` via ``asend``. ``LlmResponse`` is
            yielded with the final LLM response content on success.
            ``LlmError`` is yielded if an exception occurs during
            processing.

        Examples
        --------
        >>> processor = agent.aprompt_processor("List me the python files")
        >>> response = await anext(processor)
        >>> while isinstance(response, ActionRequest):
        ...   action = await async_input(response.description + "\napprove/reject? ")
        ...   response = await processor.asend(ApprovalDecision(action))
        >>> processor.aclose()
        >>> # process response message or error
        """

    @abstractmethod
    def change_llm(self, llm_config: LlmConfig) -> None:
        """Switch the underlying LLM and reinitialize the context config.

        Note: Changing LLM during the session is not supported, yet.

        Parameters
        ----------
        llm_config : LlmConfig
            The new LLM configuration to use for subsequent processing.
        """
        pass

    @abstractmethod
    def set_tool_policy(self, tool_name: str, policy: ToolPolicy) -> None:
        """Set the tool access policy for a specific tool.

        Parameters
        ----------
        tool_name : str
            The name of the tool whose policy is being set.
        policy : ToolPolicy
            The access policy to apply. If ``ToolPolicy.INHERIT``, the
            policy is resolved from the agent's default tool policy.

        Raises
        ------
        ValueError
            If the agent does not have a tool with the given name.
        """
        pass

    @abstractmethod
    def update_callbacks(self, callbacks: Callbacks) -> None:
        pass


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
    storage : BaseStorage or None, optional
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
        storage: BaseStorage | None = None,
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
        storage : BaseStorage or None, optional
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
