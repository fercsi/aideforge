import pytest
from pydantic import ValidationError
from typing import Literal
from unittest.mock import MagicMock

from aideforge.structs.settings import (
    BaseSettings,
    AssistantSettings,
    UiSettings,
    DisplaySettings,
    HistorySettings,
    ThinkingOption,
    SettingsGroup,
    serializable_settings,
    validate_settings,
    get_settings_type,
    _settings_registry,
)
from aideforge.structs.llm import LlmConfig
from aideforge.abc.storage import AbstractStorage


class TestThinkingOption:
    def test_values(self):
        assert ThinkingOption.OFF == "off"
        assert ThinkingOption.TEXT == "text"
        assert ThinkingOption.TOOL == "tool"
        assert ThinkingOption.ALL == "all"
        assert ThinkingOption.DETAILS == "details"


class TestDisplaySettings:
    def test_default_values(self):
        ds = DisplaySettings()
        assert ds.prompt_string == "@\\[{agent}]> "
        assert "Permission required:" in ds.tool_prompt_string
        assert ds.thinking == ThinkingOption.OFF
        assert ds.show_warnings is False

    def test_custom_values(self):
        ds = DisplaySettings(
            prompt_string="> ",
            thinking=ThinkingOption.ALL,
            show_warnings=True
        )
        assert ds.prompt_string == "> "
        assert ds.thinking == ThinkingOption.ALL
        assert ds.show_warnings is True


class TestHistorySettings:
    def test_default_values(self):
        hs = HistorySettings()
        assert hs.path is None
        assert hs.size == 1000

    def test_custom_values(self):
        hs = HistorySettings(path="/tmp/history", size=500)
        assert hs.path == "/tmp/history"
        assert hs.size == 500


class TestBaseSettings:
    def test_creation_with_storage(self):
        mock_storage = MagicMock(spec=AbstractStorage)
        settings = BaseSettings(domain="test", storage=mock_storage)
        assert settings.domain == "test"
        assert settings._storage is mock_storage

    def test_creation_without_storage(self):
        settings = BaseSettings(domain="test")
        assert settings._storage is None

    def test_set_storage(self):
        settings = BaseSettings(domain="test")
        mock_storage = MagicMock(spec=AbstractStorage)
        settings.set_storage(mock_storage)
        assert settings._storage is mock_storage

    def test_kind_version_defaults(self):
        settings = BaseSettings(domain="test")
        assert settings.kind == "settings"
        assert settings.version == 1

    def test_metadata(self):
        assert BaseSettings.storage_name == "{struct.domain}"
        assert BaseSettings.kinds == "settings"
        assert BaseSettings.type_fields == ["kind", "version", "domain"]


class TestAssistantSettings:
    def test_creation(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        settings = AssistantSettings(llm=llm)
        assert settings.domain == "assistant"
        assert settings.llm == llm
        assert isinstance(settings.history, HistorySettings)

    def test_missing_llm_raises(self):
        with pytest.raises(ValidationError):
            AssistantSettings()


class TestUiSettings:
    def test_default_values(self):
        settings = UiSettings()
        assert settings.domain == "ui"
        assert settings.theme == "default"
        assert isinstance(settings.display, DisplaySettings)

    def test_custom_values(self):
        settings = UiSettings(theme="monochrome")
        assert settings.theme == "monochrome"


class TestSettingsRegistry:
    def setup_method(self):
        # Clear registry before each test
        _settings_registry.clear()

    def test_serializable_settings_decorator(self):
        @serializable_settings("custom_domain")
        class CustomSettings(BaseSettings):
            domain: str = "custom_domain"
            custom_field: str = "default"

        assert "custom_domain" in _settings_registry
        assert _settings_registry["custom_domain"] is CustomSettings

    def test_validate_settings(self):
        @serializable_settings("test_domain")
        class TestSettings(BaseSettings):
            domain: Literal["test_domain"] = "test_domain"
            value: int = 42

        data = {"domain": "test_domain", "value": 100}
        result = validate_settings(data)
        assert isinstance(result, TestSettings)
        assert result.value == 100

    def test_validate_settings_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown settings domain: unknown"):
            validate_settings({"domain": "unknown"})

    def test_get_settings_type(self):
        @serializable_settings("another_domain")
        class AnotherSettings(BaseSettings):
            domain: Literal["another_domain"] = "another_domain"

        cls = get_settings_type("another_domain")
        assert cls is AnotherSettings

    def test_get_settings_type_unknown_raises(self):
        with pytest.raises(KeyError):
            get_settings_type("unknown")


class TestSettingsGroup:
    def setup_method(self):
        _settings_registry.clear()

    def _make_group_with_storage(self):
        mock_storage = MagicMock(spec=AbstractStorage)
        return SettingsGroup(storage=mock_storage)

    def test_register_assistant_settings(self):
        group = self._make_group_with_storage()
        llm = LlmConfig(model="gpt-4", provider="openai")
        assistant = AssistantSettings(llm=llm)
        group.register(assistant)
        assert group.has_domain("assistant")
        assert group.assistant.llm == llm

    def test_register_ui_settings(self):
        group = self._make_group_with_storage()
        ui = UiSettings(theme="monochrome")
        group.register(ui)
        assert group.has_domain("ui")
        assert group.ui.theme == "monochrome"

    def test_register_override(self):
        group = self._make_group_with_storage()
        ui1 = UiSettings(theme="default")
        ui2 = UiSettings(theme="monochrome")
        group.register(ui1)
        group.register(ui2, override=True)
        assert group.ui.theme == "monochrome"

    def test_register_ignore_failure(self):
        group = self._make_group_with_storage()
        ui1 = UiSettings(theme="default")
        ui2 = UiSettings(theme="monochrome")
        group.register(ui1)
        group.register(ui2, ignore_failure=True)
        assert group.ui.theme == "default"

    def test_register_duplicate_raises(self):
        group = self._make_group_with_storage()
        ui1 = UiSettings(theme="default")
        ui2 = UiSettings(theme="monochrome")
        group.register(ui1)
        with pytest.raises(Exception):
            group.register(ui2)

    def test_getitem_simple(self):
        group = self._make_group_with_storage()
        ui = UiSettings(theme="monochrome")
        group.register(ui)
        assert group["theme"] == "monochrome"

    def test_getitem_nested(self):
        group = self._make_group_with_storage()
        ui = UiSettings(display=DisplaySettings(prompt_string="> "))
        group.register(ui)
        assert group["display.prompt_string"] == "> "

    def test_getitem_tuple_key(self):
        group = self._make_group_with_storage()
        ui = UiSettings(display=DisplaySettings(prompt_string="> "))
        group.register(ui)
        assert group["display", "prompt_string"] == "> "

    def test_setitem_simple(self):
        group = self._make_group_with_storage()
        ui = UiSettings(theme="default")
        group.register(ui)
        group["theme"] = "monochrome"
        assert group.ui.theme == "monochrome"

    def test_setitem_nested(self):
        group = self._make_group_with_storage()
        ui = UiSettings(display=DisplaySettings(prompt_string="> "))
        group.register(ui)
        group["display.prompt_string"] = ">>> "
        assert group.ui.display.prompt_string == ">>> "

    def test_get_with_default(self):
        group = self._make_group_with_storage()
        ui = UiSettings()
        group.register(ui)
        assert group.get("theme") == "default"
        assert group.get("nonexistent") is None

    def test_get_options_top_level(self):
        group = self._make_group_with_storage()
        ui = UiSettings()
        group.register(ui)
        options = group.get_options()
        assert "theme" in options
        assert "display" in options

    def test_get_options_nested(self):
        group = self._make_group_with_storage()
        ui = UiSettings()
        group.register(ui)
        options = group.get_options("display")
        assert "prompt_string" in options

    def test_get_options_enum(self):
        group = self._make_group_with_storage()
        ui = UiSettings()
        group.register(ui)
        options = group.get_options("display.thinking")
        assert "off" in options
        assert "all" in options

    def test_has_domain(self):
        group = self._make_group_with_storage()
        assert not group.has_domain("assistant")
        llm = LlmConfig(model="gpt-4", provider="openai")
        group.register(AssistantSettings(llm=llm))
        assert group.has_domain("assistant")