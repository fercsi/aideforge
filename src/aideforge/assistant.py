import re
import shlex
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from .agent import Agent, AgentRuntime, LlmResponse, LlmError
from .command import (
    BaseCommand,
    CommandExecutor,
    CommandSuccess,
    CommandError,
    default_commands,
)

__all__ = ["Assistant", "PromptError", "PromptResult"]

PromptError = Annotated[
    CommandError | LlmError, Field(descriminator="err_code")
]

PromptResult = Annotated[
    LlmResponse | CommandSuccess | PromptError, Field(descriminator="status")
]


class Assistant:
    """AI agent and chat manager.

    Manages agent templates, model configuration, command execution, and
    the interactive chat loop. Supports switching between registered agents
    and processing user prompts (both chat and slash-commands).

    Parameters
    ----------
    model : str
        The model identifier, optionally prefixed with a provider using
        a colon (e.g. ``"openai:gpt-4o"``). If no provider is included
        and ``model_provider`` is ``None``, a :class:`ValueError` is raised.
    model_provider : str or None, optional
        The model provider name (e.g. ``"openai"``). If ``None``, the
        provider is parsed from *model* if it contains a colon.
    model_parameters : dict[str, str | int | float | None] or None, optional
        Additional parameters for model initialization (e.g. temperature).
    agents : list[Agent] or None, optional
        A list of :class:`Agent` templates to register. Defaults to a
        single default :class:`Agent` if ``None``.
    default_agent : str or None, optional
        The name of the agent to use by default. If ``None``, the first
        agent in the list is used.
    callbacks : BaseCallbackHandler or list[BaseCallbackHandler] or None, optional
        Callback handler(s) for observing LLM and tool execution.
    """

    def __init__(
        self,
        *,
        model: str,
        model_provider: str | None = None,
        model_parameters: dict[str, str | int | float | None] | None = None,
        agents: list[Agent] | None = None,
        default_agent: str | None = None,  # first agent by default
        callbacks: BaseCallbackHandler
        | list[BaseCallbackHandler]
        | None = None,
    ):
        agents = agents or [Agent()]
        self.agent_templates = {}
        for agent in agents:
            if agent.name in self.agent_templates:
                raise ValueError(
                    f"Multiple agents added with the name '{agent.name}'."
                )
            self.agent_templates[agent.name] = agent
        self._agents = {}
        self.current_agent_name = default_agent or agents[0].name
        self._current_agent = None
        self._prompt_config = None
        if callbacks:
            self.set_callbacks(callbacks)
        if model_provider is None:
            if ":" not in model:
                raise ValueError("Model provider is missing")
            self._model_provider, self._model = model.lower().split(":", 1)
        else:
            self._model_provider = model_provider.lower()
            self._model = model.lower()
        self._model_parameters = model_parameters
        # Model itself is created lazily
        self.command_executor = CommandExecutor(self)
        self.register_commands(default_commands)
        self._start_new_session()

    def _start_new_session(self) -> None:
        """Generate a new unique session ID.

        This is called on initialization and can be used to reset the
        session identifier, generating a new UUID.
        """
        self.session_id = str(uuid.uuid4())

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
        if self._prompt_config is None:
            self._prompt_config = {"callbacks": callbacks}
        elif "callbacks" not in self._prompt_config:
            self._prompt_config["callbacks"] = callbacks
        else:
            self._prompt_config["callbacks"].extend(callbacks)

    def _get_current_agent(self) -> AgentRuntime:
        """Retrieve or lazily instantiate the current agent runtime.

        If the current agent has not yet been instantiated, it is created
        from its template using the assistant's model configuration.

        Returns
        -------
        AgentRuntime
            The runtime instance for the currently selected agent.

        Raises
        ------
        KeyError
            If the current agent name does not correspond to a registered
            agent template.
        """
        name = self.current_agent_name
        if name not in self._agents:
            agent_template = self.agent_templates.get(name, None)
            if agent_template is None:
                raise KeyError(f"Agent {self.current_agent_name} not found")
            agent = agent_template.instantiate(
                model=self._model,
                model_provider=self._model_provider,
                model_parameters=self._model_parameters,
#>                config=self._prompt_config,
            )
            self._agents[name] = agent
        return self._agents[name]

    async def process_prompt(self, prompt: str) -> PromptResult:
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
        """
        prompt = prompt.strip()
        if prompt.startswith("/"):
            return self._process_command(prompt)

        agent = self._get_current_agent()
        result = await agent.process_prompt(
            prompt,
            config=self._prompt_config,
        )
        return result

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
        parts = shlex.split(prompt[1:])
        command = parts[0] if parts else ""
        args = parts[1:]
        return self.command_executor.execute_command(command, args)

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
        # {agent}           default
        # {model}           gpt-4o
        # {model_provider}  openai
        # {time:hm}         19:35
        # {time:hms}        19:35:48
        # {date:ymd}        2026-09-25
        # {date:md}         09-25
        # {input_sn}        139
        # {context_size:k}  128k
        # {token_count}     - last time? so far? all the agents...?
        # {session_id}      Session ID
        fmt = ""
        if ":" in var:
            var, fmt = var.split(":", 1)
        var = var.strip()
        fmt = fmt.strip()
        if var == "agent":
            return self.current_agent_name
#>        if var == "model":
#>            agent = self._get_current_agent()
#>        if var == "model_provider"
        if var == "time":  # local time, ISO format
            if fmt == "hms":
                return datetime.now().strftime("%H:%M:%S")
            else:
                return datetime.now().strftime("%H:%M")
        if var == "date":
            if fmt == "md":
                return datetime.now().strftime("%m-%d")
            else:
                return datetime.now().strftime("%Y:%m:%d")
        raise ValueError("Unknown prompt string variable '{var}'")

    def stop_processing(self) -> None:
        """Stop any ongoing prompt processing.

        Currently disabled.
        """
        pass
