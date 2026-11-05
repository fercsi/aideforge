from enum import StrEnum
from typing import Any, Callable, TYPE_CHECKING
from dataclasses import dataclass

from langchain.messages import SystemMessage
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field


from aideforge.structs import LlmConfig
from aideforge.tools import ToolPolicy
from aideforge.utils import IdType

__all__ = [
    "AgentSetup",
    "ActionRequest",
    "ApprovalDecision",
    "Callbacks",
    "LangChainCallbacks",
]

@dataclass
class Callbacks:
    debug: Callable[[str], None] | None = Field(
        default=None,
        description="Callback to use when some debug info should be sent to the interface",
    )
    info: Callable[[str], None] | None = Field(
        default=None,
        description="Callback to use when some info should be sent to the interface",
    )
    warning: Callable[[str], None] | None = Field(
        default=None,
        description="Callback to use when a warning should be sent to the interface",
    )


@dataclass
class LangChainCallbacks(Callbacks):
    llm: list[BaseCallbackHandler] = Field(
        default_factory=list, description="Callback handlers used by LangChain"
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
    callbacks : Callbacks or None, optional
        Callback handler to run on various points of the operation.
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
    callbacks: Callbacks | None = Field(
        default=None,
        description=(
            "Callback handler to run on various points of the operation.",
        ),
    )
