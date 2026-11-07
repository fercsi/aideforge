from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import AsyncGenerator, Generator

    from aideforge.abc.callback import AgentCallbackHandler
    from aideforge.abc.storage import AbstractStorage
    from aideforge.results import LlmError, LlmResponse
    from aideforge.structs.agent import AgentConfig
    from aideforge.structs.llm import LlmConfig
    from aideforge.structs.session import AgentState
    from aideforge.tools import ToolPolicy
    from .types import ActionRequest, ApprovalDecision

__all__ = [
    "AbstractAgentRuntime",
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
        storage: AbstractStorage | None = None,
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
        storage : AbstractStorage or None, optional
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
