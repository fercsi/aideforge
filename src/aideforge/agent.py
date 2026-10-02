# TODO decanonize/canonize model name?
__all__ = [
    "Agent",
    "AgentRuntime",
    "BaseTool",
    "Content",
    "ContextConfig",
    "LlmResponse",
    "LlmError",
]

import io
import json
import math
import os
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass, replace
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Callable, Literal

# supress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool

from .status import LlmError
from .tool import ConfigurableTool

DEFAULT_SYSTEM_PROMPT = (
    "You are my AI assistant that has no access to any tools. "
    "Please, try to answer all my questions as well as possible without using any."
)


@dataclass
class Content:
    content: str | None = None
    path: str | Path | None = None
    handle: io.TextIOBase | None = None
    type: str = "Instruction"

    def get_text(self) -> str:
        """Retrieves text content from the available source."""
        if self.content is not None:
            return self.content
        if self.path is not None:
            return Path(self.path).read_text(encoding="utf-8")
        if self.handle is not None:
            return self.handle.read()
        raise ValueError("At least one content source must be provided.")


@dataclass
class ContextConfig:
    """
    max_input_tokens is set automatically, if not set to model's maximum
    max_prompt_token is a percentage value if <= 1.0
    """

    max_input_tokens: int | None = None  # filled automatically
    max_output_tokens: int | None = None  # filled automatically
    max_prompt_tokens: int | float = 0.2


class LlmResponse(BaseModel):
    status: Literal["response"] = "response"
    content: str | None


class AgentRuntime:
    def __init__(self, setup, info, template):
        self._setup = setup
        self._info = info
        self._template = template
        self._messages = []
        for k, v in setup.items():
            setattr(self, k, v)
        self._agent = create_agent(
            model=setup["model"],
            tools=setup["tools"],
            system_prompt=setup["system_message"],
            name=setup["name"],
        )

    async def process_prompt(
        self, prompt: str, config: dict[str, Any] | None = None
    ) -> LlmResponse | LlmError:
        max_tokens = self.context_config.max_prompt_tokens
        try:
            messages = self._messages.copy()
            messages.append(HumanMessage(content=prompt))
            # TODO: System prompt is not in this!
            messages = trim_messages(
                messages,
                max_tokens=max_tokens,
                strategy="last",
                token_counter=self.model,
                include_system=False,  # keep system prompt (but not in that :(
                start_on="human",
            )

            prev_len = len(messages)
            result = await self._agent.ainvoke(
                {"messages": messages}, config=config
            )
            messages = result["messages"]
            self._messages = messages
            # self._context_size = result["messages"][-1].response_metadata["token_usage"]["total_tokens"]
            return LlmResponse(content=result["messages"][-1].content)
        except Exception as exc:
            # Many LLMs with many interfaces
            if hasattr(exc, "body") and "error" in exc.body:
                err_msg = exc.body["error"].get("message", str(exc))
            else:
                err_msg = str(exc)
            return LlmError(error_message=err_msg)

    def get(info_id:str) -> Any:
        if info_id in self._info:
            return self._info[info_id]
        if info_id in self._setup:
            return self._setup[info_id]
        if info_id[0] != "_" and hasattr(self._template, info_id):
            return getattr(self._template, info_id)
        raise KeyError(f"Unknown info ID {info_id}")


class Agent:
    name: str = "default"
    descrition: str = "General agent for common question & answer chat"
    system_prompt: SystemMessage | str = DEFAULT_SYSTEM_PROMPT
    tools: list[ConfigurableTool | BaseTool | Callable] = []
    tool_setup: dict[str, dict[str, Any]] = {}
    # +config, allow, deny, ask for each config?
    resources: list[Content] = []
    welcome_message: str = ""
    model: str | None = None
    model_provider: str | None = None
    model_parameters: dict[str, Any] | None = None
    context_config: ContextConfig = ContextConfig()
    # context settings?
    # hooks
    # TODO tool control?

    def __init__(self, setup: dict["str", Any] | None = None):
        # TODO: setup from description
        # tool: { webtool: {rights: "", config:...} }
        self.tools = self.tools.copy()
        self.resources = self.resources.copy()
        self.context_config = replace(self.context_config)

    def instantiate(
        self,
        model: str | None = None,
        model_provider: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        force_model: bool = False,
    ) -> AgentRuntime:
        setup = {
            "name": self.name,
        }
        if self.model is None or force_model:
            if model is None:
                raise TypeError(
                    "instantiate() missing required argument 'model'"
                )
        else:
            model = self.model
            model_provider = self.model_provider
            model_parameters = self.model_parameters

        setup["model"] = _get_model(model, model_provider, model_parameters)

        tools = []
        for tool in self.tools:
            config = None
            if isinstance(tool, ConfigurableTool):
                config = tool.config
                tool = tool.tool
            if isinstance(tool, type):  # incl. BaseTool
                if config is None:
                    tool = tool()
                else:
                    tool = tool(**config)
            elif not callable(tool):
                raise ValueError("Invalid tool")
            tools.append(tool)
        setup["tools"] = tools or None
        setup["context_config"] = self._init_context_config(setup)
        setup["system_message"] = self._create_system_message(setup)
        info = {
            "model": model,
            "model_provider": model_provider,
            "model_parameters": model_parameters,
        }
        return AgentRuntime(setup, info, self)

    def _create_system_message(self, setup) -> SystemMessage:
        context_config = setup["context_config"]
        system_prompt = self.system_prompt
        resources = self.resources
        if isinstance(system_prompt, SystemMessage):
            return system_prompt
        system_prompt = system_prompt.strip()
        if resources is not None and len(resources) > 0:
            for resource in resources:
                system_prompt += (
                    f"\n\n<{resource.type}>\n"
                    + resource.get_text().trim()
                    + f"\n</{resource.type}>"
                )
        max_tokens = context_config.max_prompt_tokens
        # self._context_size = max_tokens # + tool-tokens
        if max_tokens is not None:
            token_count = _get_token_count(setup["model"], system_prompt)
            if token_count > max_tokens:
                raise ValueError(
                    f"System prompt too long: {token_count} tokens exceed "
                    f"maximum limit of {max_tokens}."
                )
        return SystemMessage(system_prompt)

    def _init_context_config(self, setup) -> ContextConfig:
        cfg = replace(self.context_config)  # create a copy
        model = setup["model"]
        max_input_tokens = model.profile["max_input_tokens"]
        if cfg.max_input_tokens is None:
            cfg.max_input_tokens = max_input_tokens
        elif cfg.max_input_tokens > max_input_tokens:
            raise ValueError(
                f"context_config.max_input_tokens {cfg.max_input_tokens} "
                "exceeds the model's maximum context size."
            )
        # don't check if user set output token count, silently change it
        cfg.max_output_tokens = model.profile["max_output_tokens"]
        max_prompt_tokens = cfg.max_prompt_tokens
        if max_prompt_tokens > cfg.max_input_tokens:
            raise ValueError(
                "Maximum size of system prompt {cfg.max_prompt_tokens} "
                "exceeds the maximum context size."
            )
        elif max_prompt_tokens <= 1.0:  # percentage
            cfg.max_prompt_tokens = math.ceil(
                max_prompt_tokens * cfg.max_input_tokens
            )
        return cfg


def _get_model(model, model_provider, model_parameters):
    global _models_in_use
    if model is None:
        return None
    if not isinstance(model, str):
        return model
    # reuse model!
    model_id = hash(json.dumps((model, model_provider, model_parameters)))
    if model_id not in _models_in_use:
        model_parameters = model_parameters or {}
        model = init_chat_model(
            model=model, model_provider=model_provider, **model_parameters
        )
        _models_in_use[model_id] = model
    return _models_in_use[model_id]


_models_in_use = {}


def _get_token_count(model, text: str) -> int:
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        # Hack: HugginhFsce warnings cannot be reliably muted
        with redirect_stderr(stderr_buffer):
            token_count = model.get_num_tokens(text)

        # Warning if tokens cannot be counted exactly?!
        # if self.show_warnings:
        #     has_fallback_warning = any(
        #         "fallback" in str(w.message)
        #         for w in captured_warnings
        #     )
        #
        #     if has_fallback_warning:
        #         warning_text = (
        #            "The current model does not support native token counting. "
        #            "The returned token count is an estimate and may be inaccurate."
        #         )
        return token_count
