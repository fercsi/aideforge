__all__ = ["AgentCommand"]

from . import (
    BaseCommand,
    CommandExecutionError,
    CommandOutput,
    CommandResult,
    InvalidArgument,
    Success,
)


class AgentCommand(BaseCommand):
    name: str = "agent"
    description: str = "Perform agent related operations"

    def setup(self, parser, subparsers: dict, **kwargs) -> None:
        parser_group = parser.add_subparsers()

        subparser = parser_group.add_parser(
            "list", description="Lists all available agents"
        )
        subparser.set_defaults(func=self.list)
        subparsers["list"] = subparser

        subparser = parser_group.add_parser(
            "use", description="Selects which aget to use from now"
        )
        subparser.add_argument(
            "agent_name", metavar="AGENT_NAME", help="Agent's name to use"
        )
        subparser.set_defaults(func=self.use)
        subparsers["use"] = subparser

        subparser = parser_group.add_parser(
            "swap", description="Swaps current agent with the last one"
        )
        subparser.set_defaults(func=self.swap)
        subparsers["swap"] = subparser

        self._last_agent_name = None

    def list(self, args) -> CommandResult:
        agent_names = sorted(self._assistant.agent_templates)
        c = self._assistant.current_agent_name
        agent_names = [f"✓ {a}" if a == c else f"  {a}" for a in agent_names]
        return CommandOutput(content="\n".join(agent_names) + "\n")

    def use(self, args) -> CommandResult:
        name = args.agent_name
        if name not in self._assistant.agent_templates:
            return InvalidArgument(f"Unknown agent '{name}'")
        self._last_agent_name = self._assistant.current_agent_name
        self._assistant.current_agent_name = name
        return Success()

    def swap(self, args) -> CommandResult:
        # TODO: if only two agents, just swap
        if self._last_agent_name is None:
            return Success()
        tmp = self._last_agent_name
        self._last_agent_name = self._assistant.current_agent_name
        self._assistant.current_agent_name = tmp
        return Success()
