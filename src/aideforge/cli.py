import readline
from dataclasses import dataclass
from typing import Any, Type

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .assistant import Assistant
from .theme import Theme, DefaultTheme, to_term_colors

__all__ = ["Cli", "HistoryConfig"]

#>class VerbosityConfig


class CliCallbackHandler(BaseCallbackHandler):
    """Callback handler for printing LLM events to the terminal.

    Prints colorized information about LLM completions, tool calls, and
    errors using the colors from the configured theme.

    Parameters
    ----------
    theme : Theme
        The theme providing color definitions for terminal output.
    """

    def __init__(self, theme: Theme) -> None:
        """Initialize the callback handler with a theme.

        Parameters
        ----------
        theme : Theme
            The theme providing color definitions for terminal output.
        """
        super().__init__()
        self._theme = theme
        self._colors = to_term_colors(theme.colors)

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
        color = self._colors.thinking
        color_em = self._colors.thinking_em
        for gen in response.generations[0]:  # currently, no multiple prompts
            # there won't be multiple response variants, but who knows
            msg = gen.message
            _color_print(color_em, f"[{msg.id}] {type(msg).__name__}")
            akwa = getattr(msg, "additional_kwargs", {})
            if "reasoning_content" in akwa:
                _color_print(color, f"  Reasoning: {akwa['reasoning_content']}")
            if msg.tool_calls:
                _color_print(color, f"  Tool calls:")
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
        color = self._colors.thinking
        color_em = self._colors.thinking_em
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
        print(exception)
        print(kwargs)


@dataclass
class HistoryConfig:
    """Configuration for the readline command history.

    Attributes
    ----------
    path : str or None, optional
        File path to persist/read command history from. If ``None``,
        history is not persisted to disk.
    size : int or None, optional
        Maximum number of history entries to keep. If ``None``, the
        readline default is used.
    """

    path: str | None = None
    size: int | None = 1000


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
        theme: Theme = DefaultTheme(),
        history_config: HistoryConfig = HistoryConfig(),
        callback_handler: Type[BaseCallbackHandler] = CliCallbackHandler,
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
        self._assistant = assistant
        self._assistant.set_callbacks(callback_handler(theme=theme))
        self._theme = theme
        self._history_config = history_config

    async def run(self) -> None:
        """Run the interactive CLI loop.

        Reads user input from the terminal, displays the prompt string
        from the theme, and processes each input line via the assistant.
        History is persisted to disk if a history path is configured.

        The loop terminates on ``EOFError``, ``KeyboardInterrupt``, or
        when the assistant returns an ``"exit"`` status.
        """
        hist_cfg = self._history_config
        theme = self._theme
        colors = to_term_colors(theme.colors)
        nocolor = "\x1b[0m"

        if hist_cfg.path is not None:
            try:
                readline.read_history_file(hist_cfg.path)
            except (FileNotFoundError, OSError):
                pass
        if hist_cfg.size is not None:
            readline.set_history_length(hist_cfg.size)
        readline.set_auto_history(True)

        ps_s = colors.prompt_string
        ps_e = ""
        if colors.prompt_string:
            ps_e = nocolor
        ps_e += colors.input

        try:
            while True:
                ps_c = self._assistant.eval_prompt_string(theme.prompt_string)
                ps = ps_s + ps_c + ps_e
                try:
                    query = input(ps).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                _reset_color(colors.input)

                if not query:
                    continue

                result = await self._assistant.process_prompt(query)
                if result.status == "response":
                    _color_print(
                        colors.response, result.content, smart_end="\n"
                    )
                elif result.status == "ok":
                    pass
                elif result.status == "output":
                    _color_print(colors.output, result.content, smart_end="\n")
                elif result.status == "error":
                    _color_print(colors.output_error, result.error_message)
                elif result.status == "exit":
                    break
                else:
                    raise RuntimeError(
                        f"Command status '{result.status}' is unknown"
                    )
        finally:
            if hist_cfg.path is not None:
                try:
                    readline.write_history_file(hist_cfg.path)
                except OSError:
                    pass


def _color_print(
    color: str,
    *args: Any,
    smart_end: str | None = None,
    max_lines: int | None = None,
    indent: int = 0,
    **kwargs: Any,
) -> None:
    """Print text to the terminal with an optional color prefix.

    Parameters
    ----------
    color : str
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
    if smart_end:
        if not isinstance(args[-1], str) or not args[-1].endswith(smart_end):
            kwargs["end"] = smart_end
        else:
            kwargs["end"] = ""
    print(color, end="")
    print(*args, **kwargs)
    if color != "":
        print("\x1b[0m", end="")


def _set_color(color: str) -> None:
    """Print a color escape sequence without a newline.

    Parameters
    ----------
    color : str
        The ANSI escape sequence to print. If empty, nothing is printed.
    """
    if color:
        print(color, end="")


def _reset_color(color: str) -> None:
    """Reset the terminal color if a color was previously set.

    Parameters
    ----------
    color : str
        The color escape sequence that was previously set. If non-empty,
        the ANSI reset sequence is printed.
    """
    if color:  # not None or ""
        print("\x1b[0m", end="")
