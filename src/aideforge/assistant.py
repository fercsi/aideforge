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
    """AI agent and chat manager"""

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
        self.session_id = str(uuid.uuid4())

    def set_callbacks(
        self, callbacks: BaseCallbackHandler | list[BaseCallbackHandler]
    ) -> None:
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
        for command in commands:
            self.command_executor.register_command(command)

    def _process_command(self, prompt: str) -> PromptResult:
        parts = shlex.split(prompt[1:])
        command = parts[0] if parts else ""
        args = parts[1:]
        return self.command_executor.execute_command(command, args)

    def eval_prompt_string(self, template: str) -> str:
        ps = template.replace("{{", "\0").replace("}}", "\1")

        ps = re.sub(
            r"{([^{}]*)}",
            lambda m: self._ps_replacer(m.group(1)),
            ps,
        )

        return ps.replace("\0", "{").replace("\1", "}")

    def _ps_replacer(self, var: str) -> str:
        # {agent}           default
        # {model}           gpt-4o
        # {model_provider}  openai
        # {time:hm}         19:35
        # {time:hmss}       19:35:48
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
        pass
