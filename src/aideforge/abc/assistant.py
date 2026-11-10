from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Generator

from aideforge.agent import ActionRequest, ApprovalDecision
from aideforge.commands import BaseCommand
from aideforge.results import PromptResult
from aideforge.structs.session import SessionState
from aideforge.structs.settings import SettingsGroup
from .runtime import AbstractAgentRuntime
from .callback import AgentCallbackHandler
from .lifecycle import Lifecycle
from .storage import AbstractStorage

__all__ = ["AbstractAssistant", "TemplateResolver"]


class TemplateResolver(ABC):
    """Abstract base class for template variable resolvers.

    Implement this class to provide custom template variable resolution
    for string evaluation. The :meth:`resolve` method is called
    for each placeholder variable encountered in a template.

    WARNING: The :meth:`resolve` method must not raise exceptions when
    a variable or formatting option is unsupported. It must return
    ``None`` instead.

    Parameters
    ----------
    name : str
        Template variable name.
    fmt : str
        Template variable formatting option. Empty string (``""``) if omitted.

    Returns
    -------
    str or None
        The string value to substitute for the variable placeholder.
        Returns ``None`` if the variable or formatting option is unsupported,
        instead of raising an exception.
    """

    @abstractmethod
    def resolve(self, name: str, fmt: str) -> str | None:
        """Resolve a template variable to its string value.

        Parameters
        ----------
        name : str
            Template variable name.
        fmt : str
            Template variable formatting option. Empty string (``""``) if omitted.

        Returns
        -------
        str or None
            The string value to substitute, or ``None`` if unsupported.
        """
        ...


class AbstractAssistant(Lifecycle):
    """AI agent and chat manager.

    Manages agent blueprints, model configuration, command execution, and
    the interactive chat loop. Supports switching between registered agents
    and processing user prompts (both chat and slash-commands).
    """

    # Basic properties

    @property
    @abstractmethod
    def description(self) -> str:
        """Assistant description."""
        ...

    @property
    @abstractmethod
    def settings(self) -> SettingsGroup:
        """Assistant settings group."""
        ...

    @property
    @abstractmethod
    def storage(self) -> AbstractStorage:
        """Storage backend."""
        ...

    # Prompt processing

    @abstractmethod
    def prompt_processor(
        self,
        prompt: str,
        *,
        callbacks: AgentCallbackHandler
        | list[AgentCallbackHandler]
        | None = None,
    ) -> Generator[ActionRequest, ApprovalDecision, PromptResult]:
        """Create a processor to process a user prompt and return the result.

        If the prompt starts with ``/``, it is treated as a command and
        dispatched to the command executor. Otherwise, the prompt is sent
        to the current agent for LLM processing.

        .. warning::
           The method returns a Python *generator* that can request actions
           from the user during processing by yielding them. The caller sends
           the user's response back using ``send()``. The final result of
           the processing is the generator's return value.

        Parameters
        ----------
        prompt : str
            The user's input text.
        callbacks : AgentCallbackHandler, list[AgentCallbackHandler] or None, optional
            Callback handler(s) for progress and debug information.

        Yields
        ------
        ActionRequest
            Requests for human approval to execute a tool.

        Returns
        -------
        PromptResult
            The result of processing the prompt or command. This may be an
            LLM's response, command output or an error.

        Examples
        --------
        >>> processor = assistant.prompt_processor("List me the python files")
        >>> try:
        ...     action_req = next(processor)
        ...     while True:
        ...         action = input(action_req.description + "\napprove/reject? ")
        ...         action_req = processor.send(
        ...             ApprovalDecision.APPROVE
        ...             if action == "approve"
        ...             else ApprovalDecision.REJECT)
        ...         )
        ... except StopIteration as exc:
        ...     response = exc.value
        >>> # process response LLM message, command output or error
        """
        ...

    @abstractmethod
    async def aprompt_processor(
        self,
        prompt: str,
        *,
        callbacks: AgentCallbackHandler
        | list[AgentCallbackHandler]
        | None = None,
    ) -> AsyncGenerator[ActionRequest | PromptResult, ApprovalDecision]:
        """Create an asynchronous processor to process a user prompt and
        return the result

        If the prompt starts with ``/``, it is treated as a command and
        dispatched to the command executor. Otherwise, the prompt is sent
        to the current agent for LLM processing.

        .. warning::
           The method returns a Python *generator* that can request actions
           from the user during processing by yielding them. The caller sends
           the user's response back using ``send()``. The final result of
           the processing is also yielded (see also PEP525, "Specificaton",
           "Asynchronous Generators")

        Parameters
        ----------
        prompt : str
            The user's input text. Leading/trailing whitespace is stripped.
        callbacks : AgentCallbackHandler, list[AgentCallbackHandler] or None, optional
            Callback handler(s) for progress and debug information.

        Yields
        ------
        ActionRequest or PromptResult
            ``ActionRequest`` instances when the agent needs human approval
            to invoke a tool; ``PromptResult`` with the final result.

        Examples
        --------
        >>> processor = assistant.aprompt_processor("List me the python files")
        >>> response = await anext(processor)
        >>> while isinstance(response, ActionRequest):
        ...     action = await async_input(
        ...         response.description + "\napprove/reject? "
        ...     )
        ...     response = await processor.asend(
        ...         ApprovalDecision.APPROVE
        ...         if action == "approve"
        ...         else ApprovalDecision.REJECT
        ...     )
        >>> await processor.aclose()
        >>> # process response message or error
        """
        ...

    # Session management

    @property
    @abstractmethod
    def session(self) -> SessionState:
        """Current session state."""
        ...

    @abstractmethod
    def change_session(
        self, session: SessionState | dict[str, Any] | str | None = None
    ) -> None:
        """Change the active session.

        Parameters
        ----------
        session : SessionState, dict[str, Any], str or None, optional
            The session to switch to. Can be a :class:`SessionState` object,
            a dictionary to create one, a session ID string, or ``None`` to
            create a new session.
        """
        ...

    # Agent management

    @property
    @abstractmethod
    def available_agents(self) -> list[str]:
        """List of available agent names."""
        ...

    @property
    @abstractmethod
    def current_agent(self) -> AbstractAgentRuntime:
        """Currently active agent runtime."""
        ...

    @abstractmethod
    def change_agent(self, agent: str) -> None:
        """Change the active agent.

        Parameters
        ----------
        agent : str
            Name of the agent to switch to.

        Raises
        ------
        ValueError
            If the agent name is not registered.
        """
        ...

    # Command management

    @property
    @abstractmethod
    def commands(self) -> Mapping[str, BaseCommand]:
        """Registered commands mapping."""
        ...

    @abstractmethod
    def register_commands(self, commands: list[type[BaseCommand]]) -> None:
        """Register assistant commands.

        Parameters
        ----------
        commands : list[type[BaseCommand]]
            A list of commands (subclasses of :class:`BaseCommand`) to
            register for use as slash-commands.
        """
        ...

    # Miscellaneous functions

    @abstractmethod
    def render_template(
        self, template: str, *extensions: TemplateResolver
    ) -> str:
        """Evaluate a string template with runtime values.

        Single-brace placeholders ``{var}`` are replaced with their
        corresponding runtime values (e.g. ``{agent}``, ``{time:hm}``).
        Double braces (``{{`` and ``}}``) are preserved as literal braces
        (``{`` and ``}``) in the output.

        Supported variables include ``{agent}``, ``{model}``,
        ``{model_provider}``, ``{time:hm}``, ``{time:hms}``,
        ``{date:md}``, and ``{date:ymd}``.

        Parameters
        ----------
        template : str
            The template string containing placeholders.
        *extensions : TemplateResolver
            Additional template resolvers to extend the variable set.

        Returns
        -------
        str
            The evaluated string with all supported placeholders
            replaced.

        Raises
        ------
        ValueError
            If an unknown placeholder variable is encountered.
        """
        ...
