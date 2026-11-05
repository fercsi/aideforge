# TODO manage context info fields
from __future__ import annotations

import asyncio

# import io
import os
import warnings

# from contextlib import redirect_stderr
from threading import Lock
from typing import Any, AsyncGenerator, Awaitable, Callable

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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime as LangGraphRuntime
from langgraph.types import Command as LangGraphCommand

from aideforge.abc import AbstractAgentRuntime, AbstractStorage
from aideforge.results import LlmError, LlmResponse, InternalError
from aideforge.structs import (
    AgentConfig,
    AgentState,
    ContextConfig,
    ContextInfo,
    LlmConfig,
)
from aideforge.tools import ToolPolicy
from aideforge.utils import create_id
from .types import ActionRequest, AgentSetup, ApprovalDecision, Callbacks, LangChainCallbacks


__all__ = ["AgentRuntime"]


class AgentRuntime(AbstractAgentRuntime):
    # AbstractAgentRuntime implementation. don't repeat doc
    def __init__(
        self,
        *,
        setup: AgentSetup,
        config: AgentConfig,
        storage: AbstractStorage | None = None,
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
        storage : AbstractStorage or None, optional
            Storage backend for retrieving content resources. If ``None``,
            resources can be implemented only with their inline content.
        """
        self._setup = setup.model_copy(deep=True)
        self._config = config.model_copy(deep=True)

        self._agent_id = setup.agent_id

        if setup.threads is not None:
            self._threads = setup.threads
        else:
            self._threads = [create_id("thread")]

        if setup.tool_policy is not None:
            self._tool_policy = setup.tool_policy
        else:
            self._tool_policy = {}
            for name, tool in config.tools.items():
                self._tool_policy[name] = ToolPolicy.INHERIT
                self.set_tool_policy(name, tool.policy)

        self._state = AgentState(
            config=self._config,
            threads=self._threads,
            llm=setup.llm_config,
            tool_policy=self._tool_policy,
        )

        self.change_llm(setup.llm_config)
        self.update_callbacks(setup.callbacks)

        self._storage = storage
        if storage:
            checkpointer = storage.get_checkpoint_saver()
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
        tools = setup.tools and setup.tools.copy()
        self._agent = create_agent(
            model=self._llmodel,
            tools=tools,
            system_prompt=self._setup.system_message,
            name=self.name,
            middleware=middleware,
            checkpointer=checkpointer,
        )

        self._threads_lock = asyncio.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    def prompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> Generator[ActionRequest, ApprovalDecision, LlmResponse | LlmError]:
        raise NotImplementedError(
            "Synchron version of prompt processor is not implemented, yet"
        )

    async def aprompt_processor(
        self,
        prompt: str,
        *,
        thread_idx: int = 0,
    ) -> AsyncGenerator[
        ActionRequest | LlmResponse | LlmError, ApprovalDecision
    ]:
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
            print(self._callbacks)
            state = await self._agent.aget_state(config)
            prev_msg_cnt = (
                len(state.values["messages"])
                if "messages" in state.values
                else 0
            )
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
                    LangGraphCommand(resume={"decisions": decisions}),
                    config=config,
                )
                state = await self._agent.aget_state(config)

            ctxinfo = self._state.context_info
            ctxinfo.last_input_tokens = 0
            ctxinfo.last_output_tokens = 0
            ctxinfo.last_total_tokens = 0
            ctxinfo.last_reasoning_tokens = 0
            for msg in state.values["messages"][prev_msg_cnt:]:
                if msg.type != "ai":
                    continue
                usage = (
                    msg.usage_metadata if hasattr(msg, "usage_metadata") else {}
                )
                details = usage.get("output_token_details", {})
                ctxinfo.last_input_tokens += usage.get("input_tokens", 0)
                ctxinfo.last_output_tokens = usage.get("output_tokens", 0)
                ctxinfo.last_total_tokens = usage.get("total_tokens", 0)
                ctxinfo.last_reasoning_tokens = details.get("reasoning", 0)
            message = response["messages"][-1]
            usage = (
                message.usage_metadata
                if hasattr(message, "usage_metadata")
                else {}
            )
            details = usage.get("output_token_details", {})
            context_usage = (
                usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0)
                - details.get("reasoning", 0)
            )
            ctxinfo.current_tokens = context_usage

            yield LlmResponse(content=response["messages"][-1].content)
        except Exception as exc:
            # TODO: Somehow separate LLM and other errors:
            # Many LLMs with many interfaces: TODO work out more
            if hasattr(exc, "body") and "error" in exc.body:
                # OpenAI and compatible
                err_msg = exc.body["error"].get("message", str(exc))
            else:
                err_msg = str(exc)
#>            yield LlmError(error_message=err_msg)
            yield InternalError.from_exception(exc)

    def change_llm(self, llm_config: LlmConfig) -> None:
        if hasattr(self, "llm_config"):
            raise NotImplementedError(
                "Changing LLM during the session is not supported, yet."
            )
        self._llm_config = llm_config.model_copy()
        self._state.llm = self._llm_config
        self._llmodel = _get_model_instance(llm_config)  # LLM store is global
        self._init_context_config()

    def set_tool_policy(self, tool_name: str, policy: ToolPolicy) -> None:
        if tool_name not in self._tool_policy:
            raise ValueError(
                f"Agent does not have a tool with name '{tool_name}'"
            )
        if policy == ToolPolicy.INHERIT:
            policy = self._config.default_tool_policy
            if policy == ToolPolicy.INHERIT:
                policy = ToolPolicy.ASK  # global default is hard coded!
        self._tool_policy[tool_name] = policy

    def update_callbacks(self, callbacks: Callbacks) -> None:
        if isinstance(callbacks, LangChainCallbacks):
            self._callbacks = callbacks.llm
        else:
            self._callbacks = None
        if isinstance(callbacks, Callbacks):
            self._warning_callback = callbacks.warning
        else:
            self._warning_callback = None

    def _init_context_config(self) -> None:
        """Resolve and validate the context configuration for the agent.

        Copies the agent's :class:`ContextConfig`, fills in missing
        ``max_input_tokens`` and ``max_output_tokens`` from the model
        profile, and stores the result as ``context_info``.

        Raises
        ------
        ValueError
            If ``max_input_tokens`` exceeds the model's maximum context
            size.
        """
        ctxinfo = ContextInfo.model_validate(self._config.context.model_dump())
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
        self._state.context_info = ctxinfo

    def _limit_history(
        self, state: dict[str, Any], runtime: LangGraphRuntime
    ) -> dict[str, Any] | None:
        """Trim the message history to fit within the model's context window.

        Registered as a ``before_model`` middleware callback. Trims the
        message list using LangChain's :func:`trim_messages` helper, keeping
        the most recent messages that fit within the configured token limit.

        Parameters
        ----------
        state : dict[str, Any]
            The current agent state, containing a ``"messages"`` key.
        runtime : Langgraph.runtime.Runtime
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
                max_tokens=self._state.context_info.max_input_tokens,
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
