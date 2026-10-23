# TODO decanonize/canonize model name?
from __future__ import annotations

import asyncio
import io
import os
import warnings
from contextlib import redirect_stderr
from copy import deepcopy
from enum import StrEnum
from threading import Lock
from typing import Any, AsyncGenerator, Callable, Literal

# SUPRESS HUGGINGFACE WARNINGS
# ============================
#
# WORKAROUND: This is a known anti-pattern, as it modifies the process-wide
# environment at module import time and suppresses Transformers warnings
# globally. Unfortunately, alternative approaches have proven unreliable,
# likely due to a Hugging Face library issue. This is currently the only
# working solution that behaves consistently across all supported environments.
#
# IMPORTANT: Do not move this assignment from its current location, as its
# placement is essential for consistent behavior across all supported
# environments.
#
# See also `_get_token_count` and `_trim_text` below for more workarounds
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    before_model,
)
#>from langchain.agents.middleware.types import awrap_model_call
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field


from .status import LlmError
from .storage import BaseStorage
from .structs import AgentConfig, Content, ContextConfig, LlmConfig, ToolPolicy
from .tool import PreconfiguredTool, ToolType
from .utils import IdType, create_id, deep_merge

__all__ = [
    "AgentBlueprint",
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


class ActionRequest(BaseModel):
    name: str
    args: dict[str, Any]
    description: str


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    ALWAYS = "always"
    ALWAYS_ALL_TOOLS = "always_all_tools"
    APPROVE_FOR_RUN = "approve_for_run"
    APPROVE_ALL_TOOLS_FOR_RUN = "approve_all_tools_for_run"


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


class AgentSetup(BaseModel):
    # This is needed to store callback handler, and other callbacks
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: IdType = Field(..., description="Agent's unique ID.")
    name: str = Field(..., description="Agent's type name.")
    llm_config: LlmConfig = Field(
        ..., description="Configuration parameters for initial LLM."
    )
    tools: list[BaseTool] | None = Field(
        default=None,
        description="List of tools in format supported by LangChain `agent_create`",
    )
    system_message: SystemMessage = Field(
        ...,
        description="System message containing system prompt, including steering documents and other resources",
    )
    # TODO: Pydantic tries to resolve TypeDict (InterruptOnConfig)
    # interrupt_on: dict[str, InterruptOnConfig | bool] | None = Field(
    interrupt_on: dict[str, Any] | None = Field(
        default=None,
        description="Configuration on which tools should be interrupted",
    )
    callbacks: BaseCallbackHandler | None = Field(
        default=None,
        description="Callback handler to run on various points of the operation.",
    )
    warning_callback: Callable[str, None] | None = Field(
        default=None,
        description="Callback routine to run if a warning should be sent to user",
    )


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

    Warning: Never use an AgentRuntime instance in multiple sessions. Create
    one for each of them.

    Parameters
    ----------
    setup : dict[str, Any]
        A dictionary containing the resolved agent configuration, including
        keys such as ``"model"``, ``"tools"``, ``"system_message"``,
        ``"name"``, and ``"context_config"``.
    info : dict[str, Any]
        A dictionary of informational metadata about the agent, such as
        ``"model"``, ``"model_provider"``, and ``"model_parameters"``.
    blueprint : AgentBlueprint
        The :class:`AgentBlueprint` template from which this runtime was instantiated.
    """

    def __init__(
        self,
        *,
        setup: AgentSetup,
        config: AgentConfig,
    ) -> None:
        self.setup = setup
        self.config = config

        self.name = setup.name
        self.agent_id = setup.agent_id

        self._tool_policy = {}
        for name, tool in config.tools.items():
            self._tool_policy[name] = ToolPolicy.INHERIT
            self.set_tool_policy(name, tool.policy)

        self.change_llm(setup.llm_config)

        self._callbacks = setup.callbacks
        if isinstance(self._callbacks, BaseCallbackHandler):
            self._callbacks = [self._callbacks]

        self._threads = []

        self._warning_callback = setup.warning_callback

        checkpointer = MemorySaver()  # TODO: do we want persistent store
        middleware = [
            before_model(self._limit_history),
#>            awrap_model_call(self._route_model),
        ]
        if setup.interrupt_on:  # Tool execution control
            middleware.append(
                HumanInTheLoopMiddleware(interrupt_on=setup.interrupt_on)
            )
        self._agent = create_agent(
            model=self._llmodel,
            tools=setup.tools,
            system_prompt=self.setup.system_message,
            name=self.name,
            middleware=middleware,
            checkpointer=checkpointer,
        )

        self._threads_lock = asyncio.Lock()

    async def prompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> AsyncGenerator[
        ActionRequest | LlmResponse | LlmError, ApprovalDecision
    ]:
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

        Examples
        --------
        >>> processor = agent.prompt_processor("List me the python files")
        ... response = await anext(processor)
        ... while isinstance(response, ActionRequest)::
        ...   action = async_input(action_req.description + "\napprove or reject? ")
        ...   response = await processor.asend(action)
        >>> # process response message or error
        """
        async with self._threads_lock:  # to be sure
            if thread_idx >= len(self._threads):
                self._threads.extend(
                    [None] * (thread_idx - len(self._threads) + 1)
                )
            if self._threads[thread_idx] is None:
                self._threads[thread_idx] = create_id("thread")
            thread_id = self._threads[thread_idx]

        tool_policy = self._tool_policy.copy()
        try:
            message = HumanMessage(content=prompt)

            config = {
                "configurable": {"thread_id": str(thread_id)},
            }
            if self._callbacks:
                config["callbacks"] = self._callbacks
            response = await self._agent.ainvoke(
                {"messages": [message]}, config=config
            )
            state = await self._agent.aget_state(config)
            while state.interrupts:
                decisions = []
                for interrupt in state.interrupts:
                    for acreq in interrupt.value.get("action_requests", []):
                        tool_name = acreq["name"]
                        if (
                            tool_name not in tool_policy
                            or tool_policy[tool_name] == ToolPolicy.DENY
                        ):
                            action = "reject"
                        elif tool_policy[tool_name] == ToolPolicy.ALLOW:
                            action = "approve"
                        else:  # ToolPolicy.ASK
                            # send action request to caller
                            decision = yield ActionRequest.model_validate(acreq)
                            if not isinstance(decision, ApprovalDecision):
                                raise ValueError(
                                    f"Invalid decision response `{decision}`"
                                )

                            if decision == ApprovalDecision.ALWAYS:
                                self._tool_policy[tool_name] = ToolPolicy.ALLOW
                            elif decision == ApprovalDecision.ALWAYS_ALL_TOOLS:
                                for name in self._tool_policy:
                                    self._tool_policy[name] = ToolPolicy.ALLOW

                            if decision in (
                                ApprovalDecision.ALWAYS,
                                ApprovalDecision.APPROVE_FOR_RUN,
                            ):
                                tool_policy[tool_name] = ToolPolicy.ALLOW
                                action = "approve"
                            elif decision in (
                                ApprovalDecision.ALWAYS_ALL_TOOLS,
                                ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN,
                            ):
                                for name in tool_policy:
                                    tool_policy[name] = ToolPolicy.ALLOW
                                action = "approve"
                            else:
                                action = str(decision)
                        decisions.append({"type": action})
                response = await self._agent.ainvoke(
                    Command(resume={"decisions": decisions}), config=config
                )
                state = await self._agent.aget_state(config)
            # self._context_size = result["messages"][-1].response_metadata["token_usage"]["total_tokens"]
            yield LlmResponse(content=response["messages"][-1].content)
        except Exception as exc:
            # TODO: Somehow separate LLM and other errors:
            # Many LLMs with many interfaces: TODO work out more
            if hasattr(exc, "body") and "error" in exc.body:
                # OpenAI and compatible
                err_msg = exc.body["error"].get("message", str(exc))
            else:
                err_msg = str(exc)
            yield LlmError(error_message=err_msg)

    def change_llm(self, llm_config: LlmConfig) -> None:
        self._llm_config = llm_config
        self._llmodel = _get_model_instance(llm_config)  # LLM store is global
        self._init_context_config()

    def set_tool_policy(self, tool_name: str, policy: ToolPolicy) -> None:
        if tool_name not in self._tool_policy:
            raise ValueError(
                f"Agent does not have a tool with name '{tool_name}'"
            )
        if policy == ToolPolicy.INHERIT:
            policy = self.config.default_tool_policy
            if policy == ToolPolicy.INHERIT:
                policy = ToolPolicy.ASK  # global default is hard coded!
        self._tool_policy[tool_name] = policy

    def _init_context_config(self) -> None:
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
        cfg = self.config.context.model_copy()
        model = self._llmodel
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
        self.context_config = cfg

    def _limit_history(self, state, runtime: Runtime):
        # @before_model method
        _ = runtime
        messages = state["messages"]

        with warnings.catch_warnings(record=True) as captured_warnings:
            # Don't show warnings on falling back with token counting
            warnings.simplefilter("always")
            trimmed = trim_messages(
                messages,
                max_tokens=self.context_config.max_input_tokens,
                strategy="last",
                token_counter=self._llmodel,  # or "approximate"
                include_system=True,  # keep system prompt (but not in that :(
                start_on="human",
            )

        if len(trimmed) == len(messages):
            return None

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trimmed,
            ]
        }

    async def _route_model(self, request, handler):
        # @wrap_model_call method
        model = self._llmodel
        return await handler(request.override(model=model))

    def _warning(self, message: str) -> None:
        if self._warning_callback:
            self._warning_callback(message)


class AgentBlueprint:
    """Blueprint for creating and configuring an AI agent.

    Defines the agent's identity, model, tools, system prompt, context
    configuration, and resources. Instances are templates used to create
    :class:`AgentRuntime` objects via the :meth:`instantiate` method.

    Attributes
    ----------
    name : str
        The agent's name, used for identification and lookup.
    description : str
        A short description of the agent's purpose.
    system_prompt : SystemMessage or str
        The system prompt defining the agent's behavior.
    tools : list[PreconfiguredTool | BaseTool | Callable]
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

    def __init__(
        self,
        config: AgentConfig | dict[str, Any] | None = None,
        *,
        tool_pool: list[ToolType] | None = None,
        storage: BaseStorage | None = None,
        callbacks: BaseCallbackHandler
        | list[BaseCallbackHandler]
        | None = None,
        warning_callback: Callable[str, None] | None = None,
    ) -> None:
        """Initialize an :class:`AgentBlueprint` instance.

        Creates copies of mutable class attributes so that each instance
        has its own independent tools and resources lists.

        Parameters
        ----------
        setup : dict[str, Any] or None, optional
            Reserved for future use; currently unused. If provided, it
            will be used to create agen blueprintsvfrom agent settings
            instead of derivation in the future.
        """
        if config is None:
            self.config = AgentConfig()
        elif isinstance(config, AgentConfig):
            self.config = config.model_copy()
        elif isinstance(config, dict):
            self.config = AgentConfig.model_validate(config)
        else:
            raise ValueError(f"Invalid agent configuration data: '{config}'")
        self.name = self.config.name

        self._tools = {}
        if tool_pool is None:
            tool_pool = []
        for tool in tool_pool:
            if isinstance(tool, type) and issubclass(tool, BaseTool):
                name = tool.model_fields["name"].default
                self._tools[name] = tool
            elif isinstance(tool, PreconfiguredTool):
                self._tools[tool.name] = tool
            elif callable(tool):
                tool = StructuredTool.from_function(tool)
                self._tools[tool.name] = tool
            else:
                raise ValueError(f"Invalid tool '{tool}'")

        self._storage = storage

        self._callbacks = callbacks
        self._warning_callback = warning_callback

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
        tools = self._prepare_tools()
        setup = AgentSetup(
            agent_id=create_id("agent"),
            name=self.name,
            llm_config=self._prepare_llm_config(llm, force_llm),
            tools=tools,
            system_message=self._create_system_message(),
            interrupt_on=self._create_interrupt_on(tools),
            callbacks=self._callbacks,
        )
        return AgentRuntime(setup=setup, config=self.config)

    def _prepare_llm_config(
        self, llm: LlmConfig | None, force_llm: bool
    ) -> LlmConfig:
        if self.config.llm is None or force_llm:
            if llm is None:
                raise TypeError(
                    "instantiate() missing required argument 'model'"
                )
        else:
            llm = self.config.llm
        return llm

    def _prepare_tools(self) -> list[ToolType] | None:
        tools = []
        for name, cfg in self.config.tools.items():
            # name - user given id to tool
            # type - type name of tool
            # same 4ype tool can bevused with different config under
            # different names
            tool_type = cfg.type or name
            tool = self._tools.get(tool_type)
            if tool is None:
                self._warning(
                    f"Agent wants to use tool '{tool_type}' but tool does not exist."
                )
                # TODO: collect issues without failing
                continue
            cfg_config = None
            if cfg.config is not None:
                cfg_config = (
                    cfg.config.copy()
                )  # shallow copy is sufficient here
            if name != tool_type:
                # instantiated tool should have the name given by the user
                if cfg_config:
                    cfg_config["name"] = name
                else:
                    cfg_config = {"name": name}
            if isinstance(tool, type) and issubclass(tool, BaseTool):
                if cfg_config:
                    tool = tool.model_validate(cfg_config)
                else:
                    tool = tool()
            elif isinstance(tool, PreconfiguredTool):
                # preconfigured tool with extra configuration?
                if cfg_config and tool.config:
                    cfg_config = deep_merge(cfg_config, tool.config)
                elif tool.config:
                    cfg_config = tool.config
                if cfg_config:
                    tool = tool.tool.model_validate(cfg_config)
                else:
                    tool = tool.tool()
            else:
                raise ValueError(f"Invalid tool '{name}'")
            tools.append(tool)
        return tools or None

    def _create_system_message(self) -> SystemMessage:
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
        system_prompt = self.config.system_prompt
        resources = self.config.resources
        system_prompt = system_prompt.strip()
        if resources is not None and len(resources) > 0:
            for resource in resources:
                if isinstance(resource, str):
                    resource = Content(path=resource)
                system_prompt += (
                    f"\n\n<{resource.type}>\n"
                    + self._get_text_from_content(resource).strip()
                    + f"\n</{resource.type}>"
                )
        return SystemMessage(content=system_prompt)

    def _get_text_from_content(self, content: Content) -> str:
        if content.content is not None:
            return str(content.content)
        elif content.path is not None:
            if self._storage is None:
                raise RuntimeError(
                    f"No storage specified to retrieve {content.path}"
                )
            return self._storage.retrieve_resource(content.path)
        else:
            raise ValueError(f"Invalid content object '{content}'")

    def _create_interrupt_on(
        self, tools: list[ToolType] | None
    ) -> dict[str, InterruptOnConfig | bool] | None:
        # Currently every tool call is interrupted and approved automatically
        # by RuntimeAgent if possible. This way the interruptability can be
        # controlled during the operation. Policy settings are here not
        # considered
        if tools is None:
            return None
        config = InterruptOnConfig(
            # TODO: in case of some tools "respond" might be also a good option
            #       instead of contination with LLM
            allowed_decisions=["approve", "reject"],
            description=lambda tool_call, state, runtime: f"Tool: {tool_call['name']}\nArgs: {tool_call['args']}",
        )
        # cannot use self.config.tools, because some may fail while createing
        # runtime tools. So use runtime tool set instead
        return {t.name: deepcopy(config) for t in tools}

    def _warning(self, message: str) -> None:
        if self._warning_callback:
            self._warning_callback(message)


def _get_model_instance(llm: LlmConfig) -> BaseChatModel:
    """Initialize or retrieve a cached chat model instance.

    Parameters
    ----------
    llm : LlmConfig
        The LLM descriptor configuration model

    Returns
    -------
    BaseChatModel
        The initialized or cached chat model instance
    """
    global _models_in_use, _models_lock
    if not isinstance(llm, LlmConfig):
        raise ValueError(f"Invalid model desriptor '{llm}'")

    with _models_lock:  # to make it thread safe
        # Reuse model:
        if llm not in _models_in_use:
            _models_in_use[llm] = init_chat_model(
                model=llm.model, model_provider=llm.provider, **llm.parameters
            )
        return _models_in_use[llm]


# This cache grows over time. Since users are not expected to use more than
# a few models, memory leaks are not expected to be an issue. In the future,
# we may consider evicting models that have not been used for a long time.
_models_in_use = {}
_models_lock = Lock()


#>def _get_token_count(model: BaseChatModel, text: str) -> tuple[int, bool]:
#>    """Count the number of tokens in *text* using the given model.
#>
#>    Note: some models do not support native token counting. In this case
#>    the returned token count is an estimate and may be inaccurate.
#>
#>    Warnings from the underlying model are captured and suppressed.
#>    HuggingFace warnings that cannot be reliably muted are redirected
#>    to a buffer.
#>
#>    Parameters
#>    ----------
#>    model : BaseChatModel
#>        The language model used to count tokens.
#>    text : str
#>        The text for which to count tokens.
#>
#>    Returns
#>    -------
#>    tuple[int, bool]
#>        The number of tokens in *text* according to the model's tokenizer
#>        and a flag which is True if the value is only estimated.
#>    """
#>    stderr_buffer = io.StringIO()
#>    with warnings.catch_warnings(record=True) as captured_warnings:
#>        warnings.simplefilter("always")
#>
#>        # Hack: HuggingFace warnings cannot be reliably muted
#>        with redirect_stderr(stderr_buffer):
#>            # TODO: Synchronous call. No asynchronous alternative :(
#>            token_count = model.get_num_tokens(text)
#>
#>        # Warning if tokens cannot be counted exactly?!
#>        is_estimated = any(
#>            "fallback" in str(w.message) for w in captured_warnings
#>        )
#>        return token_count, is_estimated
#>
#>
#>def _trim_text(
#>    model: BaseChatModel, text: str, max_tokens: int
#>) -> tuple[str, bool]:
#>    curcnt, is_estimated = _get_token_count(model, text)
#>    if curcnt <= max_tokens:
#>        return text, is_estimated
#>    estimated_length = len(text) * max_tokens // curcnt
#>    return text[:estimated_length], True
#>    # stderr_buffer = io.StringIO()
#>    # with warnings.catch_warnings(record=True) as captured_warnings:
#>    #     warnings.simplefilter("always")
#>    #
#>    #     # Hack: HuggingFace warnings cannot be reliably muted
#>    #     with redirect_stderr(stderr_buffer):
#>    #         tokens = model.get_token_ids(text)
#>    #
#>    #         if len(tokens) > max_tokens:
#>    #             tokens = tokens[:max_tokens]
#>    #             text = model.decode(tokens)
#>    #
#>    #     # Warning if tokens cannot be counted exactly?!
#>    #     is_estimated = any(
#>    #         "fallback" in str(w.message) for w in captured_warnings
#>    #     )
#>    #     return text, is_estimated
