# TODO: failures in prompt/command might prevent session storage
import re
import shlex
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from functools import partial
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    TypeAlias,
)


from langchain_core.callbacks import BaseCallbackHandler
from pydantic import Field

from aideforge.abc.assistant import AbstractAssistant, TemplateResolver
from aideforge.abc.callback import AgentCallbackHandler
from aideforge.abc.command import BaseCommand
from aideforge.abc.storage import AbstractStorage
from aideforge.agent.blueprint import AgentBlueprint
from aideforge.agent.runtime import AgentRuntime
from aideforge.agent.types import ActionRequest, ApprovalDecision
from aideforge.command_executor import CommandExecutor
from aideforge.commands import default_commands
from aideforge.results import (
    CommandError, CommandSuccess, LlmError, LlmResponse, PromptError, PromptResult,
)
from aideforge.storage import MemoryStorage
from aideforge.structs.agent import AgentConfig
from aideforge.structs.llm import LlmConfig
from aideforge.structs.session import SessionState
from aideforge.structs.settings import AssistantSettings, BaseSettings, SettingsGroup, get_settings_type, validate_settings
from aideforge.structs.theme import Theme, theme_collection
from aideforge.tools import ToolType

__all__ = ["Assistant"]

DEFAULT_DESCRIPTION = (
    "I’m a versatile AI assistant that can leverage multiple specialized "
    "agents and tools to solve complex tasks. I adapt to your needs, "
    "coordinate different capabilities, and help you turn ideas into "
    "practical results."
)


AssistantSettingsInputType = (
    BaseSettings | dict[str, Any] | list[BaseSettings | dict[str, Any]]
)
AssistantAgentsInputType = list[
    AgentBlueprint | AgentConfig | dict[str, Any] | str
]


class Assistant(AbstractAssistant, TemplateResolver):
    """AI agent and chat manager.

    Manages agent blueprints, model configuration, command execution, and
    the interactive chat loop. Supports switching between registered agents
    and processing user prompts (both chat and slash-commands).

    Parameters
    ----------
    """

    def __init__(
        self,
        *,
        storage: AbstractStorage | None = None,
        default_settings: AssistantSettingsInputType | None = None,
        session: SessionState | dict[str, Any] | str | None = None,
        agents: AssistantAgentsInputType | None = None,
        tool_pool: list[ToolType] | None = None,
        llm: LlmConfig | dict[str, Any] | None = None,
        commands: list[type[BaseCommand]] | None = None,
        description: str = DEFAULT_DESCRIPTION,
    ):
        storage = storage or MemoryStorage()
        self._storage = storage

        # Setup overall settings
        self._settings = SettingsGroup(storage=storage)
        if default_settings is None:
            default_settings = [AssistantSettings()]
        elif not isinstance(default_settings, list):
            default_settings = [default_settings]
        for settings in default_settings:
            if isinstance(settings, dict):
                if "domain" not in settings:
                    settings["domain"] = "assistant"
            self._settings.register(settings)
        if not self._settings.has_domain("assistant"):
            self._settings.register(AssistantSettings())
        stored_settings = storage.list_structs(BaseSettings)
        for name in stored_settings:
            settings_type = get_settings_type(name)
            # TODO: Solve merging structs. difficulty: unchangesd fields
            # are enumerated and will not be able to merge.
            # So now, simply overwrite
            settings = storage.retrieve_struct(settings_type, name)
            self._settings.register(settings, override=True)

        # Setup initial agent blueprints
        agent_blueprints: list[
            AgentBlueprint
            | AgentConfig
            | dict[str, Any]
            | str  # agent configuration structure
            | tuple[  # agent type name
                IdType, dict[str, Any]
            ]  # agent ID and config for restoration
        ]
        agent_blueprints = deepcopy(agents) if agents else [AgentBlueprint()]
        agent_blueprints.extend(storage.list_structs(AgentConfig))
        # lazy loaded agent blueprints
        self.agent_blueprints = {}
        for blueprint in agent_blueprints:
            name = blueprint
            if isinstance(blueprint, dict):
                name = blueprint.get("name")
            elif not isinstance(
                blueprint, str
            ):  # AgentConfig or AgentBlueprint
                name = blueprint.name
            if name is None:
                raise ValueError(f"Agent defined without a name: {blueprint}")
            if name in self.agent_blueprints:
                raise ValueError(
                    f"Multiple agent blueprints added with the name '{name}'."
                )
            self.agent_blueprints[name] = blueprint
        self._agents = {}  # TODO name/ID

        self._session = None
        self._stored_input_count = 0
        self.change_session(session)

        self._tool_pool = tool_pool

        # Further setup
        llm = llm or self._settings.get("llm")
        if isinstance(llm, dict):
            llm = LlmConfig.model_validate(llm)
        elif isinstance(llm, LlmConfig):
            llm = llm.model_copy(deep=True)
        self._current_llm = llm

        self._command_executor = CommandExecutor(self)
        commands = (commands if commands is not None else []) + default_commands
        self.register_commands(commands)

        self._description = description
        self._themes = theme_collection.copy()

    @property
    def description(self) -> str:
        return self._description

    @property
    def settings(self) -> SettingsGroup:
        return self._settings

    @property
    def storage(self) -> AbstractStorage:
        return self._storage

    def prepare(self) -> None:
        self._storage.prepare()

    async def aprepare(self) -> None:
        await self._storage.aprepare()

    def cleanup(self) -> None:
        """
        It is highly recommended to call shutdown after usage

        Examples
        --------
        >>> assistant = Assistant(...)
        >>> try:
        ...     # work with assistant
        ... finally:
        ...     assistant.shutdown()
        """
        self._store_session()
        self._storage.cleanup

    async def acleanup(self) -> None:
        self._store_session()
        await self._storage.acleanup()


    # Prompt processing

    def prompt_processor(
        self,
        prompt: str,
        *,
        callbacks: AgentCallbackHandler | list[AgentCallbackHandler] | None = None
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

        prompt = prompt.strip()
        self._session.input_count += 1
        self._session.history.append(prompt)
        if prompt.startswith("/"):
            return self._process_command(prompt)

        agent = self._get_current_agent()
        response = yield from agent.prompt_processor(prompt, callbacks=callbacks)
        self._store_session()
        return response

    async def aprompt_processor(
        self,
        prompt: str,
        *,
        callbacks: AgentCallbackHandler | list[AgentCallbackHandler] | None = None
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
        prompt = prompt.strip()
        self._session.input_count += 1
        self._session.history.append(prompt)
        if prompt.startswith("/"):
            yield self._process_command(prompt)
            return

        agent = self._get_current_agent()
        processor = agent.aprompt_processor(prompt, callbacks=callbacks)
        try:
            response = await anext(processor)
            while True:
                if not isinstance(response, ActionRequest):
                    self._store_session()
                decision = yield response
                response = await processor.asend(decision)
        finally:
            await processor.aclose()

    # Session management

    @property
    def session(self) -> SessionState:
        return self._session


    def change_session(
        self, session: SessionState | dict[str, Any] | str | None = None
    ) -> None:
        # TODO: if inputs
        self._store_session()
        if session is None:
            self._session = SessionState()
        elif isinstance(session, str):  # session_id
            self._session = self._storage.retrieve_struct(SessionState, session)
        else:
            self._session = SessionState.model_validate(session)

        self._stored_input_count = self._session.input_count
        current_agent_name = None
        current_agent_id = self._session.current_agent
        for agent_id, agent_state in self._session.agents.items():
            # agent restoration is also lazy
            name = agent_state.config.name
            self.agent_blueprints[name] = (agent_id, agent_state)
            if current_agent_id == agent_id:
                current_agent_name = name
                # currently agents are stored by name internally to handle
                # it easier from user PoV
        if current_agent_name is not None:
            self.current_agent_name = current_agent_name
        else:
            self.current_agent_name = next(iter(self.agent_blueprints), None)

        # TODO: if self._session.current_agent:
        # self.current_agent_name = default_agent or agent_blueprints[0].name
        # self._current_agent = None

    # Agent management

    @property
    def available_agents(self) -> list[str]:
        return list(self.agent_blueprints)

    @property
    def current_agent(self) -> AgentRuntime:
        return self._get_current_agent()

    def change_agent(self, agent: str) -> None:
        self.current_agent_name = agent

    # Command management

    @property
    def commands(self) -> Mapping[str, BaseCommand]:
        return MappingProxyType(self._command_executor.commands)

    def register_commands(self, commands: list[type[BaseCommand]]) -> None:
        """Register assistant commands

        Parameters
        ----------
        commands : list[type[BaseCommand]]
            A list of commands (subclasses of :class:`BaseCommand`) to
            register for use as slash-commands.
        """
        for command in commands:
            self._command_executor.register_command(command)


    # Miscellaneous functions

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
        resolved = template.replace("{{", "\0").replace("}}", "\1")

        resolver = partial(self._resolver, extensions + (self,))
        resolved = re.sub(r"{([^{}]*)}", resolver, resolved)

        return resolved.replace("\0", "{").replace("\1", "}")

    def resolve(self, var: str, fmt: str) -> str | None:
        # implementing TemplateResolver.resolve
        return self._self_resolve(var, fmt)

    # PRIVATE METHODS

    def _resolver(self, resolvers: list[TemplateResolver], match: re.Match[str]) -> str:
        var = match.group(0)[1:-1]
        fmt = ""
        if ":" in var:
            var, fmt = var.split(":", 1)
        var = var.strip()
        fmt = fmt.strip()

        result = None
        for resolver in resolvers:
            result = resolver.resolve(var, fmt)
            if result is not None:
                return result
        orfmt = f" or format '{fmt}'" if fmt else ""
        raise ValueError(f"Unknown template variable '{var}'{orfmt}.")
        

    def _store_session(self) -> None:
        if self._session is None:
            return
        if self._session.input_count > self._stored_input_count:
            self._storage.store_struct(self._session)
        self._stored_input_count = self._session.input_count

    def _get_current_agent(self) -> AgentRuntime:
        """Retrieve or lazily instantiate the current agent runtime.

        If the current agent has not yet been instantiated, it is created
        from its blueprint using the assistant's model configuration.

        Returns
        -------
        AgentRuntime
            The runtime instance for the currently selected agent.

        Raises
        ------
        KeyError
            If the current agent name does not correspond to a registered
            agent blueprint.
        """
        name = self.current_agent_name
        if name not in self._agents:
            blueprint = self.agent_blueprints.get(name, None)
            if blueprint is None:
                raise KeyError(f"Agent {self.current_agent_name} not found")

            # setup blueprint lazily
            agent_id = None
            agent_state = None

            if isinstance(blueprint, tuple):  # agent to restore
                agent_id, agent_state = blueprint
                blueprint = agent_state.config

            if isinstance(blueprint, str):  # struct name in store
                blueprint = self._storage.retrieve_struct(AgentConfig, blueprint)
            if isinstance(blueprint, dict):
                blueprint = AgentConfig.model_validate(blueprint)
            if isinstance(blueprint, AgentConfig):
                blueprint = AgentBlueprint(
                    config=blueprint,
                    tool_pool=self._tool_pool,
                    storage=self._storage,
                )
                self.agent_blueprints[name] = blueprint

            if agent_state is not None:
                agent = blueprint.restore(agent_id=agent_id, state=agent_state)
            else:
                agent = blueprint.instantiate(llm=self._current_llm)
                agent_id = agent.agent_id
            self._current_llm = agent.state.llm
            self._agents[name] = agent
            # Add runtime state to session (even if previously we had one there)
            self._session.agents[agent_id] = agent.state
            self._store_session()
        return self._agents[name]

    def _process_command(self, prompt: str) -> PromptResult:
        """Parse and execute a slash-command from a prompt.

        Processes the prompt in a traditional shell-like manner: the first
        item is treated as the command, and the remaining items are passed
        as command arguments.

        Parameters
        ----------
        prompt : str
            The prompt string. The method assumes that the prompt starts
            with the ``/`` character.

        Returns
        -------
        PromptResult
            The result of the command execution.
        """
        # currently no asynchron command exevution possible
        parts = shlex.split(prompt[1:])
        command = parts[0] if parts else ""
        args = parts[1:]
        result = self._command_executor.execute_command(command, args)
        self._store_session()
        return result

    def _self_resolve(self, var: str, fmt: str) -> str | None:
        """Replace a single prompt-string placeholder variable.

        For supported variables, see :meth:`eval_prompt_string`

        Parameters
        ----------
        var : str
            The variable name extracted from the placeholder. May include
            a format specifier after a colon (e.g. ``"time:hm"``).

        Returns
        -------
        str
            The replacement value for the placeholder.

        Raises
        ------
        ValueError
            If *var* is not a recognized placeholder.
        """
        # {agent}           Agent's name
        # {model}           LL Model's name
        # {model_provider}  LLM provider's name
        # {time}            Time
        #                   "hm" - in hh:mm format (default format)
        #                   "hms" - in hh:mm:ss format
        # {date}            Date
        #                   "ymd" - YYYY-MM-DD format (default format)
        #                   "md" - MM-DD format
        # {sn}              Serial number of current input
        # {context}         Context size
        #                   "a" - automatic format: B if size <10000 (default)
        #                   "b" - in B
        #                   "k" - in kB
        #                   "%" - in % of maximum input tokens
        # {tokens}          Last run's token count
        #                   "i"/"o"/"t"/"r" - input, output (default), total, reasoning tokens
        #                   "a"/"b"/"k" - auto, byte, kbyte size
        # {session_id}      SessionState ID

        # Session specific info
        if var == "session_id" and fmt == "":
            return str(self._session.session_id)
        if var == "sn" and fmt == "":
            return str(self._session.input_count + 1)

        # Agent specific info
        if var == "agent" and fmt == "":
            return self.current_agent_name
        # NOTE: agents are prepared lazily. If you need info for PS, agent
        # will be created automatically before PS is displayed!
        # Because of the lazy load _current_llm would show a "wrong" current
        # LLM. e.g. "{model}@{agent}" shows the old model with the current
        if var == "model" and fmt == "":
            return self._get_current_agent().llm_config.model
        if var == "model_provider" and fmt == "":
            return self._get_current_agent().llm_config.provider
        if var == "context":
            ctxinfo = self._get_current_agent().context_info
            size = ctxinfo.current_tokens
            formatted = _format_size(size, fmt)
            if formatted is not None:
                return formatted
            if fmt == "%":
                if not size:
                    return "N/A"
                maxsize = ctxinfo.max_input_tokens
                return (
                    str(round(100 * size / maxsize, 1)).replace(".0", "") + "%"
                )
        if var == "tokens":
            ctxinfo = self._get_current_agent().context_info
            if fmt == "":
                fmt = "oa"
            elif len(fmt) == 1:
                if fmt in "iot":
                    fmt += "a"
                elif fmt in "abk":
                    fmt = "o" + fmt
            if len(fmt) == 2:
                size = None
                if fmt[0] == "i":
                    size = ctxinfo.last_input_tokens
                elif fmt[0] == "o":
                    size = ctxinfo.last_output_tokens
                elif fmt[0] == "t":
                    size = ctxinfo.last_total_tokens
                elif fmt[0] == "r":
                    size = ctxinfo.last_reasoning_tokens
                formatted = _format_size(size, fmt[1])
                if formatted is not None:
                    return formatted

        # Time related info
        if var == "time":  # local time, ISO format
            if fmt == "hms":
                return datetime.now().strftime("%H:%M:%S")
            elif fmt in ("hm", ""):
                return datetime.now().strftime("%H:%M")
        if var == "date":
            if fmt == "md":
                return datetime.now().strftime("%m-%d")
            elif fmt in ("ymd", ""):
                return datetime.now().strftime("%Y-%m-%d")
        orfmt = f" or format '{fmt}'" if fmt else ""
#>        raise ValueError(f"Unknown prompt string variable '{var}'{orfmt}.")

    def stop_processing(self) -> None:
        """Stop any ongoing prompt processing.

        Currently disabled.
        """
        raise NotImplementedError(
            "Stop processing current run is not implemented, yet"
        )


def _format_size(size: int | None, fmt: str) -> str | None:
    # None means, formatting refused (this formatter cannot handle it)
    if not size:
        # even if no size info, invalid formatters should be refused
        if fmt in ("a", "b", "k", ""):
            return "N/A"
        return None
    if fmt in ("a", ""):
        fmt = "b" if size < 10000 else "k"
    if fmt == "b":
        return str(size) + "B"
    if fmt == "k":
        return str(round(size / 1024, 1)).replace(".0", "") + "kB"
    return None
