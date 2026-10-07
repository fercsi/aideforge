# TODO decanonize/canonize model name?
from __future__ import annotations

import io
import json
import math
import os
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

# supress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel


from .status import LlmError
from .tool import ConfigurableTool

__all__ = [
    "Agent",
    "AgentRuntime",
    "BaseTool",
    "Content",
    "ContextConfig",
    "LlmResponse",
    "LlmError",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are my AI assistant that has no access to any tools. "
    "Please, try to answer all my questions as well as possible without using any."
)


@dataclass
class Content:
    """A container for content that can be loaded from multiple sources.

    Parameters
    ----------
    content : str or None, optional
        Direct text content. If provided, takes precedence over *path*
        and *handle*.
    path : str or Path or None, optional
        A file path to read text content from.
    handle : io.TextIOBase or None, optional
        A file-like object to read text content from.
    type : str, optional
        The type label of the content (e.g. ``"Instruction"``), used for
        tagging when embedded in a system prompt.

    Raises
    ------
    ValueError
        If :meth:`get_text` is called and no content source is available.
    """

    content: str | None = None
    path: str | Path | None = None
    handle: io.TextIOBase | None = None
    type: str = "Instruction"

    def get_text(self) -> str:
        """Retrieve text content from the available source.

        Content is read from the first available source in order of
        precedence: ``content``, ``path``, then ``handle``.

        Returns
        -------
        str
            The retrieved text content.

        Raises
        ------
        ValueError
            If none of the content sources are set.
        """
        if self.content is not None:
            return self.content
        if self.path is not None:
            return Path(self.path).read_text(encoding="utf-8")
        if self.handle is not None:
            return self.handle.read()
        raise ValueError("At least one content source must be provided.")


@dataclass
class ContextConfig:
    """Configuration for context window management of an LLM.

    Attributes
    ----------
    max_input_tokens : int or None, optional
        Maximum number of input tokens. If ``None``, filled automatically
        from the model's profile during instantiation.
    max_output_tokens : int or None, optional
        Maximum number of output tokens. If ``None``, filled automatically
        from the model's profile during instantiation.
    max_prompt_tokens : int or float, optional
        Maximum number of tokens reserved for the system prompt. If the
        value is between ``0`` and ``1`` it is treated as a percentage of
        ``max_input_tokens``, by default ``0.2``.
    """

    max_input_tokens: int | None = None  # filled automatically
    max_output_tokens: int | None = None  # filled automatically
    max_prompt_tokens: int | float = 0.2


class LlmResponse(BaseModel):
    """Represents a successful LLM response.

    Attributes
    ----------
    status : Literal["response"]
        The status of the response, always ``"response"``.
    content : str or None
        The text content returned by the LLM.
    """

    status: Literal["response"] = "response"
    content: str | None


class AgentRuntime:
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

    Parameters
    ----------
    setup : dict[str, Any]
        A dictionary containing the resolved agent configuration, including
        keys such as ``"model"``, ``"tools"``, ``"system_message"``,
        ``"name"``, and ``"context_config"``.
    info : dict[str, Any]
        A dictionary of informational metadata about the agent, such as
        ``"model"``, ``"model_provider"``, and ``"model_parameters"``.
    template : Agent
        The :class:`Agent` template from which this runtime was instantiated.
    """

    def __init__(
        self, setup: dict[str, Any], info: dict[str, Any], template: Agent
    ) -> None:
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
        """Process a user prompt through the LLM agent.

        The prompt is appended to the internal message history, which is
        then trimmed to fit within the configured token limit before being
        sent to the agent.

        Parameters
        ----------
        prompt : str
            The user's input prompt text.
        config : dict[str, Any] or None, optional
            Additional configuration passed to the LangChain agent's
            ``ainvoke`` call, by default ``None``. Currently, callbacks are
            the only configuration setting used internally (by
            :class:`Assistant`).

        Returns
        -------
        LlmResponse or LlmError
            The LLM response content on success, or an :class:`LlmError`
            if an exception occurs during processing.
        """
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

    def get(self, info_id: str) -> Any:
        """Retrieve various agent information.

        The function returns various information related to the agent's
        operation. This can include both static and dynamic information.

        Currently, it provides the following information:

        - **context_config** (``ContextConfig``):
        - **descrition (``str``):
        - **model** (``str``): 
        - **model_parameters** (``dict[str, Any]``):
        - **model_provider** (``str``):
        - **name** (``str``):
        - **resources** (``list[Content]``):
        - **system_message** (``SystemMessage``):
        - **system_prompt (``str``):
        - **tool_settings: (``dict[str, dict[str, Any]]``):
        - **tools** (``list[ConfigurableTool | BaseTool | Callable]``):
        - **welcome_message** (``str``):

        Parameters
        ----------
        info_id : str
            Information identifier to look up.

        Returns
        -------
        Any
            The value associated with *info_id*.

        Raises
        ------
        KeyError
            If *info_id* is not recognized.
        """
        if info_id in self._info:
            return self._info[info_id]
        if info_id in self._setup:
            return self._setup[info_id]
        if info_id[0] != "_" and hasattr(self._template, info_id):
            return getattr(self._template, info_id)
        raise KeyError(f"Unknown info ID {info_id}")


class Agent:
    """Blueprint for creating and configuring an AI agent.

    Defines the agent's identity, model, tools, system prompt, context
    configuration, and resources. Instances are templates used to create
    :class:`AgentRuntime` objects via the :meth:`instantiate` method.

    Attributes
    ----------
    name : str
        The agent's name, used for identification and lookup.
    descrition : str
        A short description of the agent's purpose.
    system_prompt : SystemMessage or str
        The system prompt defining the agent's behavior.
    tools : list[ConfigurableTool | BaseTool | Callable]
        A list of tools available to the agent.
    tool_setup : dict[str, dict[str, Any]]
        Configuration dictionaries for tools.
    resources : list[Content]
        Content resources (e.g. instructions) to embed in the system prompt.
    welcome_message : str
        A message displayed when the agent is first used.
    model : str or None
        The default model name for this agent. The name may also include the
        provider; in this case, they must be separated by a colon:
        "openai:gpt-4o". In this case, "model_provider" should be "None".
    model_provider : str or None
        The default model provider for this agent.
    model_parameters : dict[str, Any] or None
        Default parameters to pass when initializing the model. E.g.
        temperature
    context_config : ContextConfig
        Context window configuration for this agent.
    """

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

    def __init__(self, setup: dict[str, Any] | None = None) -> None:
        """Initialize an :class:`Agent` instance.

        Creates copies of mutable class attributes so that each instance
        has its own independent tools and resources lists.

        Parameters
        ----------
        setup : dict[str, Any] or None, optional
            Reserved for future use; currently unused. If provided, it
            will be used to create agen blueprintsvfrom agent settings
            instead of derivation in the future.
        """
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
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        model : str or None, optional
            If no model is specified in the agent blueprint, it needs to be
            specified here. By default, the agent’s own model is used, but
            if ``force_model`` is set to ``True``, this parameter overrides
            the agent’s own one.
        model_provider : str or None, optional
            The model provider (e.g. ``"openai"``). This is used when the
            ``model`` parameter is specified above.
        model_parameters : dict[str, Any] or None, optional
            Additional parameters passed to the model initializer. This is
            used when the ``model`` parameter is specified above.
        force_model : bool, optional
            If ``True``, the provided *model* arguments override the
            agent's defaults, by default ``False``.

        Returns
        -------
        AgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        TypeError
            If *model* is ``None`` and ``force_model`` is ``True`` (or the
            agent has no default model).
        ValueError
            If a tool is invalid, or if context configuration exceeds model
            limits.
        """
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

    def _create_system_message(self, setup: dict[str, Any]) -> SystemMessage:
        """Build the :class:`SystemMessage` from the prompt and resources.

        Concatenates the agent's system prompt with any configured
        resources, then validates the token count against the configured
        maximum.

        Parameters
        ----------
        setup : dict[str, Any]
            The setup dictionary containing ``"context_config"`` (with a
            ``"max_prompt_tokens"`` key) and ``"model"`` keys.

        Returns
        -------
        SystemMessage
            The constructed system message.

        Raises
        ------
        ValueError
            If the system prompt (with resources) exceeds the maximum
            allowed prompt token count.
        """
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
            token_count, is_estimated = _get_token_count(setup["model"], system_prompt)
            if token_count > max_tokens:
                raise ValueError(
                    f"System prompt too long: {token_count} tokens exceed "
                    f"maximum limit of {max_tokens}."
                )
        return SystemMessage(system_prompt)

    def _init_context_config(self, setup: dict[str, Any]) -> ContextConfig:
        """Resolve and validate the context configuration for the agent.

        Copies the agent's :class:`ContextConfig`, fills in missing
        ``max_input_tokens`` and ``max_output_tokens`` from the model
        profile, and converts percentage-based ``max_prompt_tokens`` to
        an absolute value.

        Parameters
        ----------
        setup : dict[str, Any]
            The setup dictionary containing the resolved ``"model"`` key.

        Returns
        -------
        ContextConfig
            A validated copy of the context configuration.

        Raises
        ------
        ValueError
            If ``max_input_tokens`` exceeds the model's maximum context
            size, or if ``max_prompt_tokens`` exceeds ``max_input_tokens``.
        """
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


def _get_model(
    model: str | BaseLanguageModel | None,
    model_provider: str | None,
    model_parameters: dict[str, Any] | None,
) -> Any:
    """Initialize or retrieve a cached chat model instance.

    If *model* is already a model object it is returned as-is.
    Otherwise, a unique model ID is computed from the arguments and
    the model is cached in the module-level ``_models_in_use`` dict to
    avoid re-initialization on repeated calls with the same parameters.

    Parameters
    ----------
    model : str or None
        The model name (e.g. ``"gpt-4o"``). If ``None``, returns ``None``.
        If already a model object, returned as-is.
    model_provider : str or None
        The provider name (e.g. ``"openai"``). May be ``None``.
    model_parameters : dict[str, Any] or None
        Additional keyword arguments for :func:`init_chat_model`.

    Returns
    -------
    Any
        The initialized or cached chat model instance, or ``None`` if
        *model* is ``None``.
    """
    global _models_in_use
    if model is None:
        return None
    if isinstance(model, BaseLanguageModel):
        return model
    if not isinstance(model, str):
        raise ValueError(f"Invalid model desriptor '{model}'")
    # Reuse model:
    model_id = hash(json.dumps((model, model_provider, model_parameters)))
    if model_id not in _models_in_use:
        model_parameters = model_parameters or {}
        model = init_chat_model(
            model=model, model_provider=model_provider, **model_parameters
        )
        _models_in_use[model_id] = model
    return _models_in_use[model_id]


_models_in_use = {}


def _get_token_count(model: BaseLanguageModel, text: str) -> tuple[int, bool]:
    """Count the number of tokens in *text* using the given model.

    Note: some models do not support native token counting. In this case
    the returned token count is an estimate and may be inaccurate.

    Warnings from the underlying model are captured and suppressed.
    HuggingFace warnings that cannot be reliably muted are redirected
    to a buffer.

    Parameters
    ----------
    model : BaseLanguageModel
        The language model used to count tokens.
    text : str
        The text for which to count tokens.

    Returns
    -------
    tuple[int, bool]
        The number of tokens in *text* according to the model's tokenizer
        and a flag which is True if the value is only estimated.
    """
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        # Hack: HugginhFsce warnings cannot be reliably muted
        with redirect_stderr(stderr_buffer):
            token_count = model.get_num_tokens(text)

        # Warning if tokens cannot be counted exactly?!
        is_estimated = any(
            "fallback" in str(w.message)
            for w in captured_warnings
        )
        # False means:
        return token_count, is_estimated
