# TODO manage context info fields
from __future__ import annotations

import asyncio
import io
import os
import warnings
from contextlib import redirect_stderr
from copy import deepcopy
from enum import StrEnum
from threading import Lock
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal

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
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    before_model,
)

# from langchain.agents.middleware.types import awrap_model_call
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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field


from .status import LlmError
from .storage import BaseStorage
from .structs import AgentConfig, AgentState, Content, ContextConfig, ContextInfo, LlmConfig, ToolPolicy
from .tool import PreconfiguredTool, ToolType
from .utils import IdType, create_id, deep_merge

__all__ = [
    "ActionRequest",
    "AgentBlueprint",
    "AgentRuntime",
    "ApprovalDecision",
    "BaseTool",
    "Content",
    "ContextConfig",
    "LlmError",
    "LlmResponse",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are my AI assistant that has no access to any tools. Please, try "
    "to answer all my questions as well as possible without using any."
)


class ActionRequest(BaseModel):
    """Represents a request to invoke a tool, pending human approval.

    Attributes
    ----------
    name : str
        The name of the tool to invoke.
    args : dict[str, Any]
        The arguments to pass to the tool.
    description : str
        A human-readable description of the action, used for approval prompts.
    """

    name: str
    args: dict[str, Any]
    description: str


class ApprovalDecision(StrEnum):
    """Enumeration of possible approval decisions for tool calls.

    Attributes
    ----------
    APPROVE : str
        Approve the current tool call and continue.
    REJECT : str
        Reject the current tool call and continue.
    ALWAYS : str
        Always approve this tool call until the end of session, including the
        current one.
    ALWAYS_ALL_TOOLS : str
        Always approve all tool calls until the end of session, including the
        current one.
    APPROVE_FOR_RUN : str
        Approve the current tool call for this run.
    APPROVE_ALL_TOOLS_FOR_RUN : str
        Approve the all tool calls for this run.
    """

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
    """Container for resolved agent configuration used to instantiate a runtime.

    Holds the resolved values (model config, tools, system message, etc.)
    that are needed to create an :class:`AgentRuntime`.

    Attributes
    ----------
    agent_id : IdType
        The agent's unique ID.
    name : str
        The agent's type name.
    llm_config : LlmConfig
        Configuration parameters for the initial LLM.
    tools : list[BaseTool] or None, optional
        List of tools in the format supported by LangChain.
    tool_policy : dict[str, ToolPolicy] | None
        Policy information to tools for agent restoration
    system_message : SystemMessage
        System message containing the system prompt, including steering
        documents and other resources.
    interrupt_on : dict[str, InterruptOnConfig] or None, optional
        Configuration on which tools should be interrupted.
    threads: list[IdType] | None
        List of thread IDs for agent restoration.",
    callbacks : BaseCallbackHandler or None, optional
        Callback handler to run on various points of the operation.
    warning_callback : Callable[str, None] or None, optional
        Callback routine to run if a warning should be sent to user.
    """

    # This is needed to store callback handler, and other callbacks
    model_config = ConfigDict(arbitrary_types_allowed=True)
    agent_id: IdType = Field(..., description="Agent's unique ID.")
    name: str = Field(..., description="Agent's type name.")
    llm_config: LlmConfig = Field(
        ..., description="Configuration parameters for initial LLM."
    )
    tools: list[BaseTool] | None = Field(
        default=None,
        description=(
            "List of tools in format supported by LangChain `agent_create`"
        ),
    )
    tool_policy: dict[str, ToolPolicy] | None = Field(
        default=None,
        description="Policy information to tools for agent restoration",
    )
    system_message: SystemMessage = Field(
        ...,
        description=(
            "System message containing system prompt, including "
            "steering documents and other resources"
        ),
    )
    # TODO: Pydantic tries to resolve TypeDict (InterruptOnConfig)
    # interrupt_on: dict[str, InterruptOnConfig | bool] | None = Field(
    interrupt_on: dict[str, Any] | None = Field(
        default=None,
        description="Configuration on which tools should be interrupted",
    )
    threads: list[IdType] | None = Field(
        default=None,
        description="List if thread IDs for agent restoration.",
    )
    callbacks: BaseCallbackHandler | None = Field(
        default=None,
        description=(
            "Callback handler to run on various points of the operation.",
        ),
    )
    warning_callback: Callable[str, None] | None = Field(
        default=None,
        description=(
            "Callback routine to run if a warning should be sent to user",
        ),
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

    Attributes
    ----------
    name : str
        Name of the agent's type
    agent_id : IdType
        Unique ID of agent
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
        storage: BaseStorage|None = None,
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
        storage : BaseStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.
        """
        self.setup = setup.model_copy(deep=True)
        self.config = config.model_copy(deep=True)

        self.name = setup.name
        self.agent_id = setup.agent_id

        if setup.threads is not None:
            self.threads = setup.threads
        else:
            self.threads = [create_id("thread")]

        if setup.tool_policy is not None:
            self.tool_policy = setup.tool_policy
        else:
            self.tool_policy = {}
            for name, tool in config.tools.items():
                self.tool_policy[name] = ToolPolicy.INHERIT
                self.set_tool_policy(name, tool.policy)

        self.state = AgentState(
            config=self.config,
            threads=self.threads,
            llm=setup.llm_config,
            tool_policy=self.tool_policy,
        )

        self.change_llm(setup.llm_config)

        self._callbacks = setup.callbacks
        if isinstance(self._callbacks, BaseCallbackHandler):
            self._callbacks = [self._callbacks]

        self._warning_callback = setup.warning_callback

        self._storage = storage
        if storage:
            checkpointer = storage.get_checkpointer()
        else:
            checkpointer = InMemorySaver()
        middleware = [
            before_model(self._limit_history),
            # awrap_model_call(self._aroute_model),
            # awrap_model_call does not work in LangGraph, yet
        ]
        if setup.interrupt_on:  # Tool execution control
            middleware.append(
                HumanInTheLoopMiddleware(interrupt_on=setup.interrupt_on)
            )
        self._agent = create_agent(
            model=self._llmodel,
            tools=setup.tools.copy(),
            system_prompt=self.setup.system_message,
            name=self.name,
            middleware=middleware,
            checkpointer=checkpointer,
        )

        self.threads_lock = asyncio.Lock()

    def prompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> Generator[ActionRequest, ApprovalDecision, LlmResponse | LlmError]:
        raise NotImplementedError("Synchron version of prompt processor is not implemented, yet")

    async def aprompt_processor(
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
        ...   action = async_input(response.description + "\napprove/reject? ")
        ...   response = await processor.asend(ApprovalDecision(action))
        >>> processor.aclose()
        >>> # process response message or error
        """
        async with self.threads_lock:  # to be sure
            if thread_idx >= len(self.threads):
                self.threads.extend(
                    [None] * (thread_idx - len(self.threads) + 1)
                )
            if self.threads[thread_idx] is None:
                self.threads[thread_idx] = create_id("thread")
            thread_id = self.threads[thread_idx]

        tool_policy = self.tool_policy.copy()
        try:
            message = HumanMessage(content=prompt)

            config = {
                "configurable": {"thread_id": str(thread_id)},
            }
            if self._callbacks:
                config["callbacks"] = self._callbacks
            state = await self._agent.aget_state(config)
            prev_msg_cnt = len(state.values["messages"]) if "messages" in state.values else 0
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
                                self.tool_policy[tool_name] = ToolPolicy.ALLOW
                            elif decision == ApprovalDecision.ALWAYS_ALL_TOOLS:
                                for name in self.tool_policy:
                                    self.tool_policy[name] = ToolPolicy.ALLOW

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

            ctxinfo = self.context_info
            ctxinfo.last_input_tokens = 0
            ctxinfo.last_output_tokens = 0
            ctxinfo.last_total_tokens = 0
            ctxinfo.last_reasoning_tokens = 0
#>            print(len(state.values["messages"]))
            for msg in state.values["messages"][prev_msg_cnt:]:
                if msg.type != "ai":
                    continue
                usage = msg.usage_metadata if hasattr(msg, "usage_metadata") else {}
                details = usage.get("output_token_details", {})
                ctxinfo.last_input_tokens += usage.get("input_tokens", 0)
                ctxinfo.last_output_tokens = usage.get("output_tokens", 0)
                ctxinfo.last_total_tokens = usage.get("total_tokens", 0)
                ctxinfo.last_reasoning_tokens = details.get("reasoning", 0)
            message = response["messages"][-1]
            usage = message.usage_metadata if hasattr(message, "usage_metadata") else {}
            details = usage.get("output_token_details", {})
            context_usage = usage.get("input_tokens", 0) + usage.get("output_tokens", 0) - details.get("reasoning", 0)
            ctxinfo.current_tokens = context_usage
            print(self.context_info)

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
        """Switch the underlying LLM and reinitialize the context config.

        Note: Changing LLM during the session is not supported, yet.

        Parameters
        ----------
        llm_config : LlmConfig
            The new LLM configuration to use for subsequent processing.
        """
        if hasattr(self, "llm_config"):
            raise NotImplementedError(
                "Changing LLM during the session is not supported, yet."
            )
        self.llm_config = llm_config.model_copy()
        self.state.llm = self.llm_config
        self._llmodel = _get_model_instance(llm_config)  # LLM store is global
        self._init_context_config()

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
        if tool_name not in self.tool_policy:
            raise ValueError(
                f"Agent does not have a tool with name '{tool_name}'"
            )
        if policy == ToolPolicy.INHERIT:
            policy = self.config.default_tool_policy
            if policy == ToolPolicy.INHERIT:
                policy = ToolPolicy.ASK  # global default is hard coded!
        self.tool_policy[tool_name] = policy


    def update_callbacks(self, callbacks:BaseCallbackHandler | list[BaseCallbackHandler]) -> None:
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        self._callbacks = callbacks

    def _init_context_config(self) -> None:
        """Resolve and validate the context configuration for the agent.

        Copies the agent's :class:`ContextConfig`, fills in missing
        ``max_input_tokens`` and ``max_output_tokens`` from the model
        profile, and stores the result as ``self.context_info``.

        Raises
        ------
        ValueError
            If ``max_input_tokens`` exceeds the model's maximum context
            size.
        """
        ctxinfo = ContextInfo.model_validate(self.config.context.model_dump())
        model = self._llmodel
        max_input_tokens = model.profile["max_input_tokens"]
        if ctxinfo.max_input_tokens is None:
            ctxinfo.max_input_tokens = max_input_tokens
        elif ctxinfo.max_input_tokens > max_input_tokens:
            raise ValueError(
                f"context_config.max_input_tokens {ctxinfo.max_input_tokens} "
                "exceeds the model's maximum context size."
            )
        # don't check if user set output token count, silently change it
        ctxinfo.max_output_tokens = model.profile["max_output_tokens"]
        self.context_info = ctxinfo

    def _limit_history(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Trim the message history to fit within the model's context window.

        Registered as a ``before_model`` middleware callback. Trims the
        message list using LangChain's :func:`trim_messages` helper, keeping
        the most recent messages that fit within the configured token limit.

        Parameters
        ----------
        state : dict[str, Any]
            The current agent state, containing a ``"messages"`` key.
        runtime : Runtime
            The LangGraph runtime (currently unused).

        Returns
        -------
        dict[str, Any] or None
            A state update dictionary with trimmed messages, or ``None``
            if no trimming was necessary.
        """
        # @before_model method
        _ = runtime
        messages = state["messages"]

        with warnings.catch_warnings(record=True) as captured_warnings:
            # Don't show warnings on falling back with token counting
            warnings.simplefilter("always")
            trimmed = trim_messages(
                messages,
                max_tokens=self.context_info.max_input_tokens,
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


    def _route_model(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Route a model call, overriding the model with the current one.

        Registered as a ``wrap_model_call`` middleware callback.

        Parameters
        ----------
        request : ModelRequest
            The incoming model request.
        handler : Callable[[ModelRequest], ModelResponse]
            The next handler in the middleware chain.

        Returns
        -------
        ModelCallResult
            The result from the overridden model call.
        """
        # @wrap_model_call method
        # awrap_model_call does not work in LangGraph
        model = self._llmodel
        return handler(request.override(model=model))

    async def _aroute_model(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Route a model call, overriding the model with the current one.

        Registered as a ``awrap_model_call`` middleware callback.

        Parameters
        ----------
        request : ModelRequest
            The incoming model request.
        handler : Callable[[ModelRequest], Awaitable[ModelResponse]]
            The next handler in the middleware chain.

        Returns
        -------
        ModelCallResult
            The result from the overridden model call.
        """
        # @awrap_model_call method
        # awrap_model_call does not work in LangGraph
        model = self._llmodel
        return await handler(request.override(model=model))

    def _warning(self, message: str) -> None:
        """Send a warning message to the warning callback if set.

        Parameters
        ----------
        message : str
            The warning message to send.
        """
        if self._warning_callback:
            self._warning_callback(message)


class AgentBlueprint:
    """Blueprint for creating and configuring an AI agent.

    Defines the agent's identity, model, tools, system prompt, context
    configuration, and resources. Instances are templates used to create
    :class:`AgentRuntime` objects via the :meth:`instantiate` method.


    Parameters
    ----------
    config : AgentConfig | dict[str, Any] | None, optional
        The agent configuration. May be an :class:`AgentConfig`
        instance, a dictionary to validate, or ``None`` to use defaults.
    tool_pool : list[ToolType] or None, optional
        A list of tools available to the agent. Each entry may be a
        :class:`BaseTool` subclass, a :class:`PreconfiguredTool`, or a
        callable. If ``None``, no tools are available.
    storage : BaseStorage or None, optional
        Storage backend for retrieving content resources. If ``None``,
        resources can be implemented only with their inline content.

    Attributes
    ----------
    name : str
        The agent's name, used for identification and lookup.
    config : AgentConfig
        See above
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
        config : AgentConfig | dict[str, Any] | None, optional
            The agent configuration. May be an :class:`AgentConfig`
            instance, a dictionary to validate, or ``None`` to use defaults.
        tool_pool : list[ToolType] or None, optional
            A list of tools available to the agent. Each entry may be a
            :class:`BaseTool` subclass, a :class:`PreconfiguredTool`, or a
            callable. If ``None``, no tools are available.
        storage : BaseStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.
        callbacks : BaseCallbackHandler | list[BaseCallbackHandler] or None, optional
            One or more callback handlers to run during agent operation.
        warning_callback : Callable[str, None] or None, optional
            A callable invoked with warning messages during operation.

        Raises
        ------
        ValueError
            If *config* is provided but is neither an :class:`AgentConfig`
            nor a ``dict``.
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
        llm : LlmConfig | None, optional
            If no model is configured in the agent blueprint, it must be
            specified here. By default, the agent's own model is used, but
            if ``force_llm`` is ``True``, this parameter overrides the
            agent's defaults.
        force_llm : bool, optional
            If ``True``, the provided *llm* overrides the agent's default
            model, by default ``False``.

        Returns
        -------
        AgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        TypeError
            If *llm* is ``None`` and ``force_llm`` is ``True`` (or the
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
        return AgentRuntime(setup=setup, config=self.config, storage=self._storage)

    def restore(
        self,
        agent_id: IdType,
        state: AgentState,
    ) -> AgentRuntime:
        """Create an :class:`AgentRuntime` from this agent's configuration.

        Resolves the model, tools, context config, and system message,
        then constructs an :class:`AgentRuntime` instance.

        Parameters
        ----------
        agent_id : IdType
        state : AgentState

        Returns
        -------
        AgentRuntime
            A runtime instance configured with the resolved model, tools,
            system message, and context config.

        Raises
        ------
        ValueError
            If a tool is invalid, or if context configuration exceeds model
            limits.
        """
        tools = self._prepare_tools()
        setup = AgentSetup(
            agent_id=gent_id,
            name=self.name,
            threads=state.threads,
            llm_config=state.llm,
            tools=tools,
            tool_policy=state.tool_policy,
            system_message=self._create_system_message(),
            interrupt_on=self._create_interrupt_on(tools),
            callbacks=self._callbacks,
        )
        return AgentRuntime(setup=setup, config=self.config, storage=self._storage)

    def update_callbacks(self, callbacks:BaseCallbackHandler | list[BaseCallbackHandler]) -> None:
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        self._callbacks = callbacks

    def _prepare_llm_config(
        self, llm: LlmConfig | None, force_llm: bool
    ) -> LlmConfig:
        """Resolve which LLM configuration to use for the agent.

        By default agent rutime uses the agent blueprint's LLM configuration.
        If it is not specified or ``force_llm`` is ``True``, ``llm`` input
        will be used instead.

        Parameters
        ----------
        llm : LlmConfig | None
            Optionally provided LLM configuration to override the agent's
            default.
        force_llm : bool
            If ``True``, use *llm* even if the agent has a default model.

        Returns
        -------
        LlmConfig
            The resolved LLM configuration.

        Raises
        ------
        TypeError
            If no LLM configuration is available.
        """
        if self.config.llm is None or force_llm:
            if llm is None:
                raise TypeError(
                    "instantiate() missing required argument 'model'"
                )
        else:
            llm = self.config.llm
        return llm

    def _prepare_tools(self) -> list[ToolType] | None:
        """Instantiate and configure tools for the agent runtime.

        Iterates over the configured tool definitions, looks up each tool
        type in the tool pool, applies per-tool configuration, and resolves
        custom names. Tools that are not found emit a warning and are
        skipped.

        Returns
        -------
        list[ToolType] or None
            A list of resolved tool instances, or ``None`` if no tools
            are configured.
        """
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
                    f"Agent wants to use tool '{tool_type}' but tool "
                    "does not exist."
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
        resources. Resources may be provided as inline content, file
        paths, or :class:`Content` objects.

        Returns
        -------
        SystemMessage
            The constructed system message.

        Raises
        ------
        RuntimeError
            If a resource references a path but no storage backend was
            provided to the blueprint.
        ValueError
            If a resource has neither inline content nor a path.
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
        """Retrieve text content from a :class:`Content` object.

        Returns the inline content if present, otherwise retrieves it from
        storage using the content's path.

        Parameters
        ----------
        content : Content
            A :class:`Content` object with either inline content or a
            storage path.

        Returns
        -------
        str
            The retrieved text content.

        Raises
        ------
        RuntimeError
            If *content* has a path but no storage backend was provided.
        ValueError
            If *content* has neither inline content nor a path.
        """
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
        """Create interrupt-on configuration for the agent's tools.

        Each tool call is interrupted and approved (or rejected) per-call
        by the runtime. This method generates the per-tool interrupt
        configuration that enables that flow. Note that per-tool policy
        settings are not considered here.

        Parameters
        ----------
        tools : list[ToolType] or None
            The resolved list of tool instances.

        Returns
        -------
        dict[str, InterruptOnConfig | bool] or None
            A mapping from tool name to interrupt configuration, or ``None``
            if no tools are provided.
        """
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
            description=lambda tool_call, state, runtime: (
                f"Tool: {tool_call['name']}\nArgs: {tool_call['args']}"
            ),
        )
        # cannot use self.config.tools, because some may fail while createing
        # runtime tools. So use runtime tool set instead
        return {t.name: deepcopy(config) for t in tools}

    def _warning(self, message: str) -> None:
        """Send a warning message to the warning callback if set.

        Parameters
        ----------
        message : str
            The warning message to send.
        """
        if self._warning_callback:
            self._warning_callback(message)


def _get_model_instance(llm: LlmConfig) -> BaseChatModel:
    """Initialize or retrieve a cached chat model instance.

    A model with a specific parameter set is created once and it will be
    reused, if the same is requested.

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

# TEMPORARILY REMOVED
# def _get_token_count(model: BaseChatModel, text: str) -> tuple[int, bool]:
#     """Count the number of tokens in *text* using the given model.
#
#     Note: some models do not support native token counting. In this case
#     the returned token count is an estimate and may be inaccurate.
#
#     Warnings from the underlying model are captured and suppressed.
#     HuggingFace warnings that cannot be reliably muted are redirected
#     to a buffer.
#
#     Parameters
#     ----------
#     model : BaseChatModel
#         The language model used to count tokens.
#     text : str
#         The text for which to count tokens.
#
#     Returns
#     -------
#     tuple[int, bool]
#         The number of tokens in *text* according to the model's tokenizer
#         and a flag which is True if the value is only estimated.
#     """
#     stderr_buffer = io.StringIO()
#     with warnings.catch_warnings(record=True) as captured_warnings:
#         warnings.simplefilter("always")
#
#         # Hack: HuggingFace warnings cannot be reliably muted
#         with redirect_stderr(stderr_buffer):
#             # TODO: Synchronous call. No asynchronous alternative :(
#             token_count = model.get_num_tokens(text)
#
#         # Warning if tokens cannot be counted exactly?!
#         is_estimated = any(
#             "fallback" in str(w.message) for w in captured_warnings
#         )
#         return token_count, is_estimated
#
#
# def _trim_text(
#     model: BaseChatModel, text: str, max_tokens: int
# ) -> tuple[str, bool]:
#     """Trim a text to a maximum number of tokens
#
#     Parameters
#     ----------
#     model : BaseChatModel
#         The language model used to count tokens.
#     text : str
#         The text for which to count tokens.
#     max_tokens : int
#         The maximum number of tokens the output text should contain
#
#     Returns
#     -------
#     tuple[str, bool]
#         The trimmed text according to the model's tokenizer and a flag
#         which is True if the value is only estimated.
#     """
#     curcnt, is_estimated = _get_token_count(model, text)
#     if curcnt <= max_tokens:
#         return text, is_estimated
#     estimated_length = len(text) * max_tokens // curcnt
#     return text[:estimated_length], True
#     # stderr_buffer = io.StringIO()
#     # with warnings.catch_warnings(record=True) as captured_warnings:
#     #     warnings.simplefilter("always")
#     #
#     #     # Hack: HuggingFace warnings cannot be reliably muted
#     #     with redirect_stderr(stderr_buffer):
#     #         tokens = model.get_token_ids(text)
#     #
#     #         if len(tokens) > max_tokens:
#     #             tokens = tokens[:max_tokens]
#     #             text = model.decode(tokens)
#     #
#     #     # Warning if tokens cannot be counted exactly?!
#     #     is_estimated = any(
#     #         "fallback" in str(w.message) for w in captured_warnings
#     #     )
#     #     return text, is_estimated
