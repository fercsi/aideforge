# TODO: failures in prompt/command might prevent session storage
import re
import shlex
from copy import deepcopy
from datetime import datetime
from typing import Annotated, Any, AsyncGenerator, Callable, Generator

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import Field

from .agent import (
    ActionRequest,
    AgentBlueprint,
    AgentRuntime,
    ApprovalDecision,
    LlmError,
    LlmResponse,
)
from .command import (
    BaseCommand,
    CommandExecutor,
    CommandSuccess,
    CommandError,
    default_commands,
)
from .storage import BaseStorage, MemoryStorage
from .structs import AgentConfig, LlmConfig, SessionState
from .tool import ToolType

__all__ = ["Assistant", "PromptError", "PromptResult"]

PromptError = Annotated[
    CommandError | LlmError, Field(descriminator="error_code")
]

PromptResult = Annotated[
    LlmResponse | CommandSuccess | PromptError, Field(descriminator="status")
]


class Assistant:
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
        storage: BaseStorage | None = None,
        session: SessionState | dict[str, Any] | str | None = None,
        agents: list[AgentBlueprint | AgentConfig | dict[str, Any] | str]
        | None = None,
        tool_pool: list[ToolType] | None = None,
        llm: LlmConfig | None = None,
        llm_callbacks: BaseCallbackHandler
        | list[BaseCallbackHandler]
        | None = None,
        commands: list[type[BaseCommand]] | None = None,
        warning_callback: Callable[str, None] | None = None,
    ):
        storage = storage or MemoryStorage()
        self._storage = storage

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
        self._agent_blueprints = {}
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
            if name in self._agent_blueprints:
                raise ValueError(
                    f"Multiple agent blueprints added with the name '{name}'."
                )
            self._agent_blueprints[name] = blueprint
        self._agents = {}  # TODO name/ID

        self.session = None
        self._stored_input_count = 0
        self.change_session(session)

        self._tool_pool = tool_pool

        # Further setup
        self._current_llm = llm
        self._llm_callbacks = []
        if llm_callbacks:
            self.set_callbacks(llm_callbacks)

        self.command_executor = CommandExecutor(self)
        commands = (commands if commands is not None else []) + default_commands
        self.register_commands(commands)

        self._warning_callback = warning_callback

    def shutdown(self) -> None:
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

    def _store_session(self) -> None:
        if self.session is None:
            return
        if self.session.input_count > self._stored_input_count:
            self._storage.store_struct(self.session)
        self._stored_input_count = self.session.input_count

    def change_session(
        self, session: SessionState | dict[str, Any] | str | None = None
    ) -> None:
        # TODO: if inputs
        self._store_session()
        if session is None:
            self.session = SessionState()
        elif isinstance(session, str):  # session_id
            self.session = self._storage.retrieve_struct(SessionState, session)
        else:
            self.session = SessionState.model_validate(session)

        self._stored_input_count = self.session.input_count
        current_agent_name = None
        current_agent_id = self.session.current_agent
        for agent_id, agent_state in self.session.agents.items():
            # agent restoration is also lazy
            name = agent_state.config.name
            self._agent_blueprints[name] = (agent_id, agent_state)
            if current_agent_id == agent_id:
                current_agent_name = name
                # currently agents are stored by name internally to handle
                # it easier from user PoV
        if current_agent_name is not None:
            self.current_agent_name = current_agent_name
        else:
            self.current_agent_name = next(iter(self._agent_blueprints), None)

        # TODO: if self.session.current_agent:
        # self.current_agent_name = default_agent or agent_blueprints[0].name
        # self._current_agent = None

    def set_callbacks(
        self, callbacks: BaseCallbackHandler | list[BaseCallbackHandler]
    ) -> None:
        """Register or extend callback handlers for LLM/tool execution.

        Callbacks are stored in the prompt configuration and applied to
        every LLM call. If callbacks were already set, new handlers are
        appended to the existing list.

        Parameters
        ----------
        callbacks : BaseCallbackHandler or list[BaseCallbackHandler]
            One or more callback handlers to register.
        """
        if isinstance(callbacks, list):
            callbacks = callbacks.copy()
        else:
            callbacks = [callbacks]
        self._llm_callbacks.extend(callbacks)
        for agent in self._agents.values():
            agent.update_callbacks(callbacks)
        for blueprint in self._agent_blueprints.values():
            if isinstance(blueprint, AgentBlueprint):
                blueprint.update_callbacks(callbacks)

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
            blueprint = self._agent_blueprints.get(name, None)
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
                    callbacks=self._llm_callbacks,
                    warning_callback=self._warning_callback,
                )
                self._agent_blueprints[name] = blueprint

            if agent_state is not None:
                agent = blueprint.restore(agent_id=agent_id, state=agent_state)
            else:
                agent = blueprint.instantiate(llm=self._current_llm)
                agent_id = agent.agent_id
            self._current_llm = agent.llm_config
            self._agents[name] = agent
            # Add runtime state to session (even if previously we had one there)
            self.session.agents[agent_id] = agent.state
            self._store_session()
        return self._agents[name]

    def prompt_processor(
        self,
        prompt: str,
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
        self.session.input_count += 1
        self.session.history.append(prompt)
        if prompt.startswith("/"):
            return self._process_command(prompt)

        agent = self._get_current_agent()
        response = yield from agent.prompt_processor(prompt)
        self._store_session()
        return response

    async def aprompt_processor(
        self,
        prompt: str,
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
        >>> processor = agent.aprompt_processor("List me the python files")
        >>> response = await anext(processor)
        >>> while isinstance(response, ActionRequest):
        ...   action = async_input(response.description + "\napprove/reject? ")
        ...   response = await processor.asend(ApprovalDecision(action))
        >>> await processor.aclose()
        >>> # process response message or error
        """
        prompt = prompt.strip()
        self.session.input_count += 1
        self.session.history.append(prompt)
        if prompt.startswith("/"):
            yield self._process_command(prompt)
            return

        agent = self._get_current_agent()
        processor = agent.aprompt_processor(prompt)
        try:
            response = await anext(processor)
            while True:
                if not isinstance(response, ActionRequest):
                    self._store_session()
                decision = yield response
                response = await processor.asend(decision)
        finally:
            await processor.aclose()

    def register_commands(self, commands: list[type[BaseCommand]]) -> None:
        """Register assistant commands

        Parameters
        ----------
        commands : list[type[BaseCommand]]
            A list of commands (subclasses of :class:`BaseCommand`) to
            register for use as slash-commands.
        """
        for command in commands:
            self.command_executor.register_command(command)

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
        result = self.command_executor.execute_command(command, args)
        self._store_session()
        return result

    def eval_prompt_string(self, template: str) -> str:
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
        ps = template.replace("{{", "\0").replace("}}", "\1")

        ps = re.sub(
            r"{([^{}]*)}",
            lambda m: self._ps_replacer(m.group(1)),
            ps,
        )

        return ps.replace("\0", "{").replace("\1", "}")

    def _ps_replacer(self, var: str) -> str:
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
        fmt = ""
        if ":" in var:
            var, fmt = var.split(":", 1)
        var = var.strip()
        fmt = fmt.strip()

        # Session specific info
        if var == "session_id" and fmt == "":
            return str(self.session.session_id)
        if var == "sn" and fmt == "":
            return str(self.session.input_count + 1)

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
        raise ValueError(f"Unknown prompt string variable '{var}'{orfmt}.")

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
