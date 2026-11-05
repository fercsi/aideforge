from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from langchain.agents.middleware import InterruptOnConfig
from langchain.messages import SystemMessage
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool

from aideforge.storage import BaseStorage
from aideforge.structs import AgentConfig, AgentState, Content, LlmConfig
from aideforge.tools import PreconfiguredTool, ToolType
from aideforge.utils import IdType, create_id, deep_merge
from .abc import AbstractAgentBlueprint
from .runtime import AgentRuntime
from .types import AgentSetup

__all__ = ["AgentBlueprint"]


class AgentBlueprint(AbstractAgentBlueprint):
    # AbstractAgentBlueprint implementation. don't repeat doc
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
        if config is None:
            self._config = AgentConfig()
        elif isinstance(config, AgentConfig):
            self._config = config.model_copy()
        elif isinstance(config, dict):
            self._config = AgentConfig.model_validate(config)
        else:
            raise ValueError(f"Invalid agent configuration data: '{config}'")

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
        return AgentRuntime(
            setup=setup, config=self._config, storage=self._storage
        )

    def restore(
        self,
        agent_id: IdType,
        state: AgentState,
    ) -> AgentRuntime:
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
        return AgentRuntime(
            setup=setup, config=self._config, storage=self._storage
        )

    def update_callbacks(
        self, callbacks: BaseCallbackHandler | list[BaseCallbackHandler]
    ) -> None:
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        self._callbacks = callbacks

    # PRIVATE METHODS

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
        if self._config.llm is None or force_llm:
            if llm is None:
                raise TypeError(
                    "instantiate() missing required argument 'model'"
                )
        else:
            llm = self._config.llm
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
        for name, cfg in self._config.tools.items():
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
        system_prompt = self._config.system_prompt
        resources = self._config.resources
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
        # cannot use self._config.tools, because some may fail while createing
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
