import pytest
from pydantic import ValidationError

from aideforge.structs.tool import ToolModeConfig, ToolConfig
from aideforge.tools import ToolPolicy


class TestToolModeConfig:
    def test_default_values(self):
        tmc = ToolModeConfig()
        assert tmc.type is None
        assert tmc.policy == ToolPolicy.INHERIT
        assert tmc.config is None

    def test_custom_values(self):
        tmc = ToolModeConfig(
            type="shell",
            policy=ToolPolicy.ALLOW,
            config={"show_errors": True}
        )
        assert tmc.type == "shell"
        assert tmc.policy == ToolPolicy.ALLOW
        assert tmc.config == {"show_errors": True}

    def test_nested_commands(self):
        tmc = ToolModeConfig(
            policy=ToolPolicy.ASK,
            config={
                "commands": {
                    "ls": ToolModeConfig(policy=ToolPolicy.ALLOW),
                    "rm": ToolModeConfig(policy=ToolPolicy.ASK),
                }
            }
        )
        assert tmc.config["commands"]["ls"].policy == ToolPolicy.ALLOW
        assert tmc.config["commands"]["rm"].policy == ToolPolicy.ASK


class TestToolConfig:
    def test_creation(self):
        tc = ToolConfig(type="test_tool")
        assert tc.kind == "tool"
        assert tc.version == 1
        assert tc.type == "test_tool"
        assert tc.config is None
        assert tc.policy == ToolPolicy.INHERIT

    def test_with_policy_and_config(self):
        tc = ToolConfig(
            policy=ToolPolicy.ALLOW,
            config={"show_errors": False}
        )
        assert tc.policy == ToolPolicy.ALLOW
        assert tc.config == {"show_errors": False}

    def test_metadata(self):
        assert ToolConfig.scope == "global"
        assert ToolConfig.storage_name == "{struct.type}"
        assert ToolConfig.kinds == "tools"

    def test_inherits_tool_mode_config(self):
        tc = ToolConfig(
            type="shell",
            policy=ToolPolicy.DENY,
            config={"commands": {"ls": {"policy": "allow"}}}
        )
        assert tc.type == "shell"
        assert tc.policy == ToolPolicy.DENY
        assert "commands" in tc.config
