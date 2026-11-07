from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Generator

from aideforge.agent import ActionRequest, AgentRuntime, ApprovalDecision
from aideforge.commands import BaseCommand
from aideforge.results import PromptResult
from aideforge.structs.session import SessionState
from aideforge.structs.settings import SettingsGroup
from .callback import AgentCallbackHandler
from .lifecycle import Lifecycle
from .storage import AbstractStorage

__all__ = ["AbstractAssistant", "TemplateResolver"]


class TemplateResolver(ABC):
    """Template rendering resolver

    Function contract for processing additional variables during prompt
    string evaluation.

    WARNING: The :meth:`resolve` method must not raise exceptions when
    a variable or formatting option is unsupported. It must return
    ``None`` instead.

    Signature
    ---------
    ``(var: str, fmt: str) -> str | None``

    Parameters
    ----------
    var : str
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
        pass


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
        ...

    @property
    @abstractmethod
    def settings(self) -> SettingsGroup:
        ...

    @property
    @abstractmethod
    def storage(self) -> AbstractStorage:
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
        """
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
        """Process a user prompt and return the result.

        If the prompt starts with ``/``, it is treated as a command and
        dispatched to the command executor. Otherwise, the prompt is sent
        to the current agent for LLM processing.

        Parameters
        ----------
        prompt : str
            The user's input text. Leading/trailing whitespace is stripped.

        Returns
        -------
        PromptResult
            The result of processing the prompt or command. This may be an
            LLM's response, command output or an error.
        Examples
        --------
        >>> processor = assistant.aprompt_processor("List me the python files")
        >>> response = await anext(processor)
        >>> while isinstance(response, ActionRequest):
        ...   action = await async_input(response.description + "\napprove/reject? ")
        ...   response = await processor.asend(ApprovalDecision(action))
        >>> await processor.aclose()
        >>> # process response message or error
        """
        ...

    # Session management

    @property
    @abstractmethod
    def session(self) -> SessionState:
        ...

    @abstractmethod
    def change_session(
        self, session: SessionState | dict[str, Any] | str | None = None
    ) -> None:
        ...

    # Agent management

    @property
    @abstractmethod
    def available_agents(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def current_agent(self) -> AgentRuntime:
        ...

    @abstractmethod
    def change_agent(self, agent: str) -> None:
        ...

    # Command management

    @property
    @abstractmethod
    def commands(self) -> Mapping[str, BaseCommand]:
        ...

    @abstractmethod
    def register_commands(self, commands: list[type[BaseCommand]]) -> None:
        """Register assistant commands

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
        """Evaluate a prompt string template with runtime values.

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

        Returns
        -------
        str
            The evaluated prompt string with all supported placeholders
            replaced.

        Raises
        ------
        ValueError
            If an unknown placeholder variable is encountered.
        """
        ...
