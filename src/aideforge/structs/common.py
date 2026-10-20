from enum import StrEnum

__all__ = ["Switch", "ToolPolicy"]


class Switch(StrEnum):
    OFF = "off"
    ON = "on"


class ToolPolicy(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
