# CLI module
# Note: CLI submodule is not thread safe, because it is built on readline,
# which isn't. But I don't think it could be an issue, ever.
import argparse
import asyncio
import json
import re
import readline
import sys
from functools import partial
from typing import Any, Callable, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from .agent import ActionRequest, ApprovalDecision, Callbacks, LangChainCallbacks
from .assistant import Assistant, PSEvalExtender
from .structs import AssistantSettings, Theme, ThemeColor, UiSettings
from .structs.theme import to_term_colors

__all__ = ["Cli", "CliCallbackHandler"]

class _Shared(BaseModel):
    colors: ThemeColor = Field(default_factory=ThemeColor)
    ui_settings: UiSettings = Field(default_factory=UiSettings)


class CliCallbackHandler(BaseCallbackHandler):
    """Callback handler for printing LLM events to the terminal.

    Prints colorized information about LLM completions, tool calls, and
    errors using the colors from the configured theme.

    Parameters
    ----------
    theme : Theme
        The theme providing color definitions for terminal output.
    """

    def __init__(self, shared: _Shared) -> None:
        """Initialize the callback handler with a theme.

        Parameters
        ----------
        theme : Theme
            The theme providing color definitions for terminal output.
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
        color = self._shared.colors.thinking
        color_em = self._shared.colors.thinking_em
        assert len(response.generations) > 0, "on_llm_end: Response does not have generations"
        for gen in response.generations[0]:  # currently, no multiple prompts
            # there won't be multiple response variants, but who knows
            msg = gen.message
#>            _color_print(color_em, f"[{msg.id}] {type(msg).__name__}")
#>            akwa = getattr(msg, "additional_kwargs", {})
#>            if "reasoning_content" in akwa:
#>                _color_print(color, f"  Reasoning: {akwa['reasoning_content']}")
            if msg.tool_calls:
                _color_print(color, "  Tool calls:")
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                _color_print(color, f"    [{tc['id']}] {tc['name']}({args})")

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
        return
        color = self._shared.colors.thinking
        color_em = self._shared.colors.thinking_em
        _color_print(
            color_em,
            f"[{output.tool_call_id}] {type(output).__name__} -> {output.name}",
        )
        _color_print(color, output.content, max_lines=3, indent=2)

    def on_llm_error(self, exception: BaseException, **kwargs: Any) -> None:
        """Handle LLM errors by printing the exception and context.

        Parameters
        ----------
        exception : BaseException
            The exception raised by the LLM call.
        **kwargs : Any
            Additional keyword arguments passed by the callback framework.
        """
        return
        print(exception)
        print(kwargs)


class Cli:
    """Command-line interface for interacting with an :class:`Assistant`.

    Provides an interactive REPL loop that reads user input, dispatches
    it to the assistant, and prints the response using the configured theme.

    Parameters
    ----------
    assistant : Assistant
        The assistant instance used to process prompts and commands.
    theme : Theme, optional
        The theme for terminal colors and prompt strings, by default
        :class:`DefaultTheme`.
    history_config : HistoryConfig, optional
        Configuration for command history persistence, by default
        :class:`HistoryConfig`.
    callback_handler : Type[BaseCallbackHandler], optional
        The callback handler class to use for observing LLM execution,
        by default :class:`CliCallbackHandler`.
    """

    def __init__(
        self,
        *,
        assistant: Assistant,
        default_settings: UiSettings|dict[str,Any]|None=None,
        callback_handler: type[BaseCallbackHandler] = CliCallbackHandler,
    ) -> None:
        """Initialize the CLI with an assistant and configuration.

        Parameters
        ----------
        assistant : Assistant
            The assistant instance used to process prompts and commands.
        theme : Theme, optional
            The theme for terminal colors and prompt strings, by default
            :class:`DefaultTheme`.
        history_config : HistoryConfig, optional
            Configuration for command history persistence, by default
            :class:`HistoryConfig`.
        callback_handler : Type[BaseCallbackHandler], optional
            The callback handler class to use for observing LLM execution,
            by default :class:`CliCallbackHandler`.
        """
        default_settings = default_settings or UiSettings()
        self._shared=_Shared()
        self._assistant = assistant
        self._storage = assistant.storage
        # TODO warning callback
        callbacks = LangChainCallbacks(llm = [callback_handler(shared=self._shared)])
        self._assistant.update_callbacks(callbacks)
        self._assistant.ensure_settings(default_settings)

    def process_arguments(self, args: list[str] | None = None, *, custom_parser:Callable[[argparse.ArgumentParser], None]|None=None) -> None:
        # Call this function if you would like to use default argument parsing
        # this function is not called automatically
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
        raise NotImplementedError("Synchron CLI run has not been implemented, yet")

    async def arun(self) -> None:
        """Run the interactive CLI loop.

        Reads user input from the terminal, displays the prompt string
        from the theme, and processes each input line via the assistant.
        History is persisted to disk if a history path is configured.

        The loop terminates on ``EOFError``, ``KeyboardInterrupt``, or
        when the assistant returns an ``"exit"`` status.
        """
        assistant = self._assistant
        session = assistant.session

        ui_settings = assistant.get_settings("ui")
        assert isinstance(ui_settings, UiSettings), f"Expected UiSettings, got {type(ui_settings).__name__}"
        assistant_settings = assistant.get_settings("assistant")
        assert isinstance(assistant_settings, AssistantSettings), f"Expected AssistantSettings, got {type(assistant_settings).__name__}"

#>        theme = self._storage.retrieve_struct(Theme, ui_settings.theme)
        theme = assistant.get_theme(ui_settings.theme)
        colors = to_term_colors(theme.colors)
        hist_size = assistant_settings.history.size
        prompt_string = ui_settings.display.prompt_string

        self._shared.ui_settings = ui_settings
        self._shared.colors = colors

        readline.clear_history()
        history = assistant.session.history
        if hist_size is not None:
            readline.set_history_length(hist_size)
        for item in history:
            readline.add_history(item)
        readline.set_auto_history(True)

        welcome = assistant.get_welcome_message()
        if welcome:
            _color_print(colors.welcome_message, welcome)

        try:
            while True:
                ps_text = assistant.eval_prompt_string(prompt_string)
                _color_print(colors.prompt_string, ps_text, end="")
                _set_color(colors.input)
                try:
                    query = await ainput("")
                    query = query.strip()
                except (EOFError, KeyboardInterrupt):
                    break
                _reset_color(colors.input)

                if not query:
                    continue

                processor = assistant.aprompt_processor(query)
                response = await anext(processor)
                while isinstance(response, ActionRequest):
                    decision = await self._aaction_input(response)
                    response = await processor.asend(decision)
                await processor.aclose()

                if response.status == "response":
                    _color_print(
                        colors.response, response.content, smart_end="\n"
                    )
                elif response.status == "ok":
                    pass
                elif response.status == "output":
                    _color_print(colors.output, response.content, smart_end="\n")
                elif response.status == "error":
                    _color_print(colors.output_error, "\n".join(response.traceback))
                    _color_print(colors.output_error, response.error_message)
                elif response.status == "exit":
                    break
                else:
                    raise RuntimeError(
                        f"Command status '{response.status}' is unknown"
                    )
        finally:
            assistant.shutdown()
            print(f"\nSession {assistant.session.session_id} has ended.")

    def _action_input(self, request: ActionRequest) -> ApprovalDecision:
        colors = self._shared.colors
        display = self._shared.ui_settings.display
        extender: PSEvalExtender = partial(_ps_extender, request)
        ps_text = self._assistant.eval_prompt_string(display.tool_prompt_string, extender)
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
        colors = self._shared.colors
        display = self._shared.ui_settings.display
        extender:PSEvalExtender = partial(_ps_extender, request)
        ps_text = self._assistant.eval_prompt_string(display.tool_prompt_string, extender)
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

def _ps_extender(request: ActionRequest, var: str, fmt: str = "") -> str|None:
    # if var or fmt is not valid, None must be returned without raising an error
    # see details at ``PSEvalExtender``
    # {tool}
    # {arga:min} = {args}
    # {args:pretty}
    # {args:pretty,4} does not indenr first row. User can indent it with whatever they want
    # {args:json}
    # {args:json,2}
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





def _color_print(
    color: str|None,
    *args: Any,
    smart_end: str | None = None,
    max_lines: int | None = None,
    indent: int = 0,
    **kwargs: Any,
) -> None:
    """Print text to the terminal with an optional color prefix.

    Parameters
    ----------
    color : str or None
        The ANSI escape sequence for the desired color. Pass an empty string
        for no color.
    *args : Any
        Positional arguments passed to :func:`print`.
    smart_end : str or None, optional
        If provided, appends this string to the output only if the last
        positional argument does not already end with it, by default
        ``None``.
    max_lines : int or None, optional
        If provided, truncates the output to this many lines, by default
        ``None``. (Currently unused — reserved for future use.)
    indent : int, optional
        Number of two-space indentation units to add, by default ``0``.
        (Currently unused — reserved for future use.)
    **kwargs : Any
        Additional keyword arguments passed to :func:`print`.

    Notes
    -----
    The color escape sequence is printed before the content and reset
    after, unless *color* is an empty string.
    """
    color = color or ""
    if smart_end:
        if not isinstance(args[-1], str) or not args[-1].endswith(smart_end):
            kwargs["end"] = smart_end
        else:
            kwargs["end"] = ""
    print(color, end="")
    print(*args, **kwargs)
    if color != "":
        print("\x1b[0m", end="")


def _set_color(color: str|None) -> None:
    """Print a color escape sequence without a newline.

    Parameters
    ----------
    color : str or None
        The ANSI escape sequence to print. If empty, nothing is printed.
    """
    if color:
        print(color, end="")


def _reset_color(color: str|None) -> None:
    """Reset the terminal color if a color was previously set.

    Parameters
    ----------
    color : str or None
        The color escape sequence that was previously set. If non-empty,
        the ANSI reset sequence is printed.
    """
    if color:  # not None or ""
        print("\x1b[0m", end="")


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)
