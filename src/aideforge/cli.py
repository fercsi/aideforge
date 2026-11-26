# CLI module
# Note: CLI submodule is not thread safe, because it is built on readline,
# which isn't. But I don't think it could be an issue, ever.
import argparse
import asyncio
import json
import re
import readline
import signal
import sys
from functools import partial
from typing import Any, Callable, cast
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from aideforge.abc.callback import AgentCallbackHandler
from aideforge.abc.terminal import Terminal
from aideforge.agent import ActionRequest, ApprovalDecision
from aideforge.assistant import Assistant, TemplateResolver
from aideforge.structs.settings import AssistantSettings, UiSettings, SettingsGroup, ThinkingOption
from aideforge.structs.theme import Theme#, ThemeColor, to_term_colors
from aideforge.terminal.rich import RichTerminal

__all__ = ["Cli", "CliCallbackHandler"]

@dataclass
class _Shared:
    terminal: Terminal | None = None
    settings: SettingsGroup = field(default_factory=SettingsGroup)


class CliCallbackHandler(AgentCallbackHandler, BaseCallbackHandler):
    """Callback handler for printing LLM events to the terminal.

    Prints colorized information about LLM completions, tool calls, and
    errors using the colors from the configured theme.

    Parameters
    ----------
    shared : _Shared
        Shared state containing theme colors and UI settings.
    """
#>    OFF = "off"
#>    TEXT = "text"
#>    TOOL = "tool"
#>    ALL = "all"
#>    DETAILS = "details"
#>    settings["display.thinking

    def __init__(self, shared: _Shared) -> None:
        """Initialize the callback handler with shared state.

        Parameters
        ----------
        shared : _Shared
            Shared state containing theme colors and UI settings.
        """
        super().__init__()
        self._shared = shared

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle the end-of-LLM-call event by printing colorized output.

        Prints the message ID and type, any reasoning content, and tool
        calls for each generation in the response.

        Parameters
        ----------
        response : LLMResult
            The result object returned by the LLM, containing one or more
            generations.
        **kwargs : Any
            Additional keyword arguments passed by the callback framework.
        """
        thinking = self._shared.settings["display.thinking"]
        if thinking == ThinkingOption.OFF:
            return
        terminal = self._shared.terminal
        assert terminal is not None, "on_llm_end: terminal has not been initialized."
        show_text = thinking in (ThinkingOption.TEXT, ThinkingOption.ALL, ThinkingOption.DETAILS)
        show_tool = thinking in (ThinkingOption.TOOL, ThinkingOption.ALL, ThinkingOption.DETAILS)
        show_details = thinking == ThinkingOption.DETAILS

        assert len(response.generations) > 0, "on_llm_end: Response does not have generations."
        for gen in response.generations[0]:  # currently, no multiple prompts
            # there won't be multiple response variants, but who knows
            terminal.print()
            msg = gen.message
            if show_details:
                terminal.print(f"{type(msg).__name__} <{msg.id}>", style="llm_id")
            if show_text:
                akwa = getattr(msg, "additional_kwargs", {})
                if "reasoning_content" in akwa:
                    terminal.print(f"Thinking: {akwa['reasoning_content']}", style="llm_thinking")
            if show_tool and msg.tool_calls:
                terminal.print("Tool calls:", style="llm_tool_call")
                for tc in msg.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    idtxt = f"[{tc['id']}] " if show_details else ""
                    terminal.print(f"  {idtxt}{tc['name']}({args})", style="llm_tool_call")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Handle the end-of-tool-execution event by printing colorized output.

        Prints the tool call ID, output type/name, and limited content.

        Parameters
        ----------
        output : Any
            The output produced by the tool.
        **kwargs : Any
            Additional keyword arguments passed by the callback framework.
        """
        thinking = self._shared.settings["display.thinking"]
        if thinking != ThinkingOption.DETAILS:
            return
        terminal = self._shared.terminal
        assert terminal is not None, "on_tool_end: terminal has not been initialized."
        terminal.print()
        terminal.print(
            f"{type(output).__name__} <{output.tool_call_id}>",
            style="llm_id",
        )
        terminal.print(
            f"Output of {output.name}:",
            style="llm_tool_output",
        )
        content = output.content
        if not isinstance(content, str):
            content = json.dumps(content)
        maxlen = terminal.width * 3 - 1
        if len(content) > maxlen:
            content = content[:maxlen-3] + "..."
        lines = content.split("\n", 3)
        if len(lines) > 3:
            content = "\n".join(lines[:3]) + "\n..."
        terminal.print(content, style="llm_tool_output")

    def on_llm_error(self, exception: BaseException, **kwargs: Any) -> None:
        """Handle LLM errors by printing the exception and context.

        Parameters
        ----------
        exception : BaseException
            The exception raised by the LLM call.
        **kwargs : Any
            Additional keyword arguments passed by the callback framework.
        """
        # Traceback settings?
        # kwargs?
        terminal = self._shared.terminal
        assert terminal is not None, "on_llm_error: terminal has not been initialized."
        terminal.print(f"LLM ERROR: {exception}", style="warning")

    # AgentCallbackHandler callbacks

    def on_warning(self, content: str) -> None:
        if not self._shared.settings["display.show_warnings"]:
            return
        terminal = self._shared.terminal
        assert terminal is not None, "on_warning: terminal has not been initialized."
        terminal.print(f"WARNING: {content}", style="warning")


class Cli:
    """Command-line interface for interacting with an :class:`Assistant`.

    Provides an interactive REPL loop that reads user input, dispatches
    it to the assistant, and prints the response using the configured theme.

    Parameters
    ----------
    assistant : Assistant
        The assistant instance used to process prompts and commands.
    default_settings : UiSettings, dict[str, Any] or None, optional
        Default UI settings for the assistant, by default UiSettings().
    callback_handler : type[BaseCallbackHandler], optional
        The callback handler class to use for observing LLM execution,
        by default :class:`CliCallbackHandler`.
    terminal : Terminal or None, optional
        The terminal instance to use for input and output. If not provided,
        a RichTerminal is created.
    """

    def __init__(
        self,
        *,
        assistant: Assistant,
        default_settings: UiSettings|dict[str,Any]|None=None,
        callback_handler: type[BaseCallbackHandler] = CliCallbackHandler,
        terminal: Terminal | None = None
    ) -> None:
        default_settings = default_settings or UiSettings()
        self._shared=_Shared()
        self._assistant = assistant
        self._storage = assistant.storage
        # TODO warning callback
        self._callbacks = callback_handler(shared=self._shared)
        # Don't override already loaded UI settings only ensure if it does not exist
        self._assistant.settings.register(default_settings, ignore_failure=True)
        self._terminal = terminal or RichTerminal()

    def process_arguments(self, args: list[str] | None = None, *, custom_parser:Callable[[argparse.ArgumentParser], None]|None=None) -> None:
        """Parse command-line arguments for the CLI.

        Call this function if you would like to use default argument parsing.
        This function is not called automatically.

        Parameters
        ----------
        args : list[str] or None, optional
            Command-line arguments to parse. Defaults to ``sys.argv``.
        custom_parser : Callable[[argparse.ArgumentParser], None] or None, optional
            Optional function to add custom arguments to the parser.
        """
        args = args or sys.argv
        parser = argparse.ArgumentParser(
            description=self._assistant.description
        )

        parser.add_argument('-s', '--session', help="Continue with the given session.")
        parser.add_argument('-v', '--verbose', action='store_true')

        if custom_parser is not None:
            custom_parser(parser)

        self._opts = parser.parse_args()
        if self._opts.session:
            self._assistant.change_session(self._opts.session)

    def run(self) -> None:
        """Run the synchronous CLI loop.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("Synchron CLI run has not been implemented, yet")

    async def arun(self) -> None:
        """Run the interactive CLI loop.

        Reads user input from the terminal, displays the prompt string
        from the theme, and processes each input line via the assistant.
        History is persisted to disk if a history path is configured.

        The loop terminates on ``EOFError``, ``KeyboardInterrupt``, or
        when the assistant returns an ``"exit"`` status.
        """
        # Disable Ctrl+C
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: None)

        assistant = self._assistant
        await assistant.aprepare()
        session = assistant.session
        settings = assistant.settings

        ui_settings = assistant.settings.ui
        assert isinstance(ui_settings, UiSettings), f"Expected UiSettings, got {type(ui_settings).__name__}"
        assistant_settings = assistant.settings.assistant
        assert isinstance(assistant_settings, AssistantSettings), f"Expected AssistantSettings, got {type(assistant_settings).__name__}"

        terminal = self._terminal

        theme = settings.get_theme()
        terminal.set_theme(theme)
#>        hist_size = assistant_settings.history.size
        prompt_string = ui_settings.display.prompt_string

        self._shared.settings = settings
        self._shared.terminal = self._terminal

        history = assistant.session.history
        terminal.set_history(history)

        welcome = assistant.current_agent.state.config.welcome_message
        if welcome:
            terminal.print(welcome, style="welcome_message")

        try:
            while True:
                ps_text = assistant.render_template(prompt_string)
                try:
                    query = await terminal.ainput(ps_text, style="prompt_string")
                    query = query.strip()
                # except (EOFError, KeyboardInterrupt):
                except EOFError:
                    break

                if not query:
                    continue

                processor = assistant.aprompt_processor(query, callbacks=self._callbacks)
                response = await anext(processor)
                while isinstance(response, ActionRequest):
                    decision = await self._aaction_input(response)
                    response = await processor.asend(decision)
                await processor.aclose()

                if response.status == "response":
                    terminal.print(
                        response.content, smart_end="\n", style="response",
                    )
                elif response.status == "ok":
                    pass
                elif response.status == "output":
                    terminal.print(response.content, smart_end="\n", style="output")
                elif response.status == "error":
                    terminal.print(response.error_message, style="output_error")
                elif response.status == "exit":
                    break
                else:
                    raise RuntimeError(
                        f"Command status '{response.status}' is unknown"
                    )
        finally:
            await assistant.acleanup()
            print(f"\nSession {assistant.session.session_id} has ended.")

    def _action_input(self, request: ActionRequest) -> ApprovalDecision:
        """Prompt the user for approval of a tool action.

        Parameters
        ----------
        request : ActionRequest
            The action request containing tool name and arguments.

        Returns
        -------
        ApprovalDecision
            The user's approval decision.
        """
        colors = self._shared.colors
        display = self._shared.ui_settings.display
        extension = PsActionResolver(request)
        ps_text = self._assistant.render_template(display.tool_prompt_string, extension)
        ps_err = None
        _color_print(colors.tool_prompt_string, ps_text, end="")
        while True:
            _set_color(colors.tool_input)
            answer = input("")
            _reset_color(colors.tool_input)
            histlen = readline.get_current_history_length()
            if histlen > 0:
                readline.remove_history_item(histlen - 1)
            decision = _eval_answer(answer)
            if decision is not None:
                return decision
            if ps_err is None:
                ps_err = self._assistant.eval_prompt_string(display.tool_prompt_error_string, extender)
            _color_print(colors.tool_prompt_string, ps_err, end="")

    async def _aaction_input(self, request: ActionRequest) -> ApprovalDecision:
        """Prompt the user asynchronously for approval of a tool action.

        Parameters
        ----------
        request : ActionRequest
            The action request containing tool name and arguments.

        Returns
        -------
        ApprovalDecision
            The user's approval decision.
        """
        colors = self._shared.colors
        display = self._shared.ui_settings.display
        extension = PsActionResolver(request)
        ps_text = self._assistant.render_template(display.tool_prompt_string, extension)
        ps_err = None
        _color_print(colors.tool_prompt_string, ps_text, end="")
        while True:
            _set_color(colors.tool_input)
            answer = await ainput("")
            _reset_color(colors.tool_input)
            histlen = readline.get_current_history_length()
            if histlen > 0:
                readline.remove_history_item(histlen - 1)
            decision = _eval_answer(answer)
            if decision is not None:
                return decision
            if ps_err is None:
                ps_err = self._assistant.eval_prompt_string(display.tool_prompt_error_string, extender)
            _color_print(colors.tool_prompt_string, ps_err, end="")

class PsActionResolver(TemplateResolver):
    """Template resolver for action approval prompts.

    Resolves template variables for tool name and arguments in the
    tool approval prompt string.

    Parameters
    ----------
    request : ActionRequest
        The action request containing the tool name and arguments.
    """

    def __init__(self, request: ActionRequest) -> None:
        self._request = request

    def resolver(self, var: str, fmt: str = "") -> str|None:
        """Resolve a template variable for the action prompt.

        Parameters
        ----------
        var : str
            The variable name to resolve (e.g., "tool", "args").
        fmt : str, optional
            Format specifier for the variable (e.g., "pretty", "json,2"),
            by default "".

        Returns
        -------
        str or None
            The resolved value, or None if the variable or format is not recognized.
        """
        # if var or fmt is not valid, None must be returned without raising an error
        # {tool}
        # {arga:min} = {args}
        # {args:pretty}
        # {args:pretty,4} does not indenr first row. User can indent it with whatever they want
        # {args:json}
        # {args:json,2}
        request = self._request
        if var == "tool":
            if fmt == "":
                return request.name
        if var == "args":
            if fmt in ("min", ""):
                return str(request.args)
            fmts = fmt.split(",")
            if len(fmts) == 1:
                indent = 0
            elif len(fmts) == 2 and fmts[1].isdigit():
                indent = int(fmts[1])
            else:
                return None
            if fmts[0] == "pretty":
                sep = "\n" + " " * indent
                argstxt = sep.join(f"{k} = {v!r}" for k, v in request.args.items())
                return argstxt
            if fmts[0] == "json":
                # TODO: ending "}" is on col 0
                argstxt = json.dumps(request.args, indent=indent)
                return argstxt
        return None

def _eval_answer(answer: str) -> ApprovalDecision | None:
    """Evaluate the user's answer to an action approval prompt.

    Parameters
    ----------
    answer : str
        The user's input string.

    Returns
    -------
    ApprovalDecision or None
        The corresponding approval decision, or None if the input
        is not recognized.
    """
    answer = answer.strip().lower()
    if answer in ("y", "yes", "approve"):
        return ApprovalDecision.APPROVE
    if answer in ("n", "no", "reject"):
        return ApprovalDecision.REJECT
    if answer in ("a", "always"):
        return ApprovalDecision.ALWAYS
    answer = re.sub(r'[\W\d_]+', '', answer)
    if answer in ("aat", "alwaysall", "alwaysalltools"):
        return ApprovalDecision.ALWAYS_ALL_TOOLS
    if answer in ("yfr", "yesforrun", "approveforrun"):
        return ApprovalDecision.APPROVE_FOR_RUN
    if answer in ("yatfr", "yesallforrun", "yestoalltoolsforrun"):
        return ApprovalDecision.APPROVE_ALL_TOOLS_FOR_RUN
    return None
