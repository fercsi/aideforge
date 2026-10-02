__all__ = ("Cli", "HistoryConfig")
import readline
from dataclasses import dataclass
from typing import Any, Type

from langchain_core.callbacks import BaseCallbackHandler

from .assistant import Assistant
from .theme import Theme, DefaultTheme, to_term_colors

#>class VerbosityConfig

class CliCallbackHandler(BaseCallbackHandler):
    def __init__(self, theme):
        super().__init__()
        self._theme = theme
        self._colors = to_term_colors(theme.colors)

    def on_llm_end(self, response, **kwargs):
        color = self._colors.thinking
        color_em = self._colors.thinking_em
        for gen in response.generations[0]: # currently, no multiple prompts
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


    def on_tool_end(self, output, **kwargs):
        color = self._colors.thinking
        color_em = self._colors.thinking_em
        _color_print(color_em, f"[{output.tool_call_id}] {type(output).__name__} -> {output.name}")
        _color_print(color, output.content, max_lines=3, indent=2)

    def on_llm_error(self, exception: BaseException, **kwargs):
        print(exception)
        print(kwargs)

@dataclass
class HistoryConfig:
    path:str=None
    size:int|None=1000

class Cli:
    def __init__(self, *,
            assistant: Assistant,
            theme: Theme = DefaultTheme(),
            history_config: HistoryConfig = HistoryConfig(),
            callback_handler: Type[BaseCallbackHandler] = CliCallbackHandler
        ):
        self._assistant = assistant
        self._assistant.set_callbacks(callback_handler(theme=theme))
        self._theme = theme
        self._history_config = history_config

    async def run(self):
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
                    _color_print(colors.response, result.content, smart_end="\n")
                elif result.status == "ok":
                    pass
                elif result.status == "output":
                    _color_print(colors.output, result.content, smart_end="\n")
                elif result.status == "error":
                    _color_print(colors.output_error, result.error_message)
                elif result.status == "exit":
                    break
                else:
                    raise RuntimeError(f"Command status '{result.status}' is unknown")
        finally:
            if hist_cfg.path is not None:
                try:
                    readline.write_history_file(hist_cfg.path)
                except OSError:
                    pass


def _color_print(color:str, *args: Any, smart_end=None, max_lines=None, indent=0, **kwargs: Any) -> None:
    if smart_end:
        if not isinstance(args[-1], str) or not args[-1].endswith(smart_end):
            kwargs["end"] = smart_end
        else:
            kwargs["end"] = ""
    print(color, end="")
    print(*args, **kwargs)
    if color != "":
        print("\x1b[0m", end="")

def _set_color(color):
    if color:
        print(color, end="")

def _reset_color(color):
    # sets color removal if color has been set previously
    if color: # not None or ""
        print("\x1b[0m", end="")
