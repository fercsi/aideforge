import pytest
from pydantic import ValidationError

from aideforge.structs.llm import LlmConfig


class TestLlmConfig:
    def test_creation_with_all_fields(self):
        llm = LlmConfig(model="gpt-4", provider="openai", parameters={"temperature": 0.7})
        assert llm.kind == "llm"
        assert llm.version == 1
        assert llm.model == "gpt-4"
        assert llm.provider == "openai"
        assert llm.parameters == {"temperature": 0.7}

    def test_creation_with_default_parameters(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        assert llm.parameters == {}

    def test_model_string_parsing(self):
        llm = LlmConfig(model="openai:gpt-4")
        assert llm.model == "gpt-4"
        assert llm.provider == "openai"

    def test_model_string_parsing_with_explicit_provider(self):
        llm = LlmConfig(model="openai:gpt-4", provider="anthropic")
        assert llm.model == "openai:gpt-4"
        assert llm.provider == "anthropic"

    def test_model_string_missing_provider_raises(self):
        with pytest.raises(ValidationError):
            LlmConfig(model="gpt-4")

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            LlmConfig(provider="openai")

    def test_hashable(self):
        llm1 = LlmConfig(model="gpt-4", provider="openai", parameters={"temperature": 0.7})
        llm2 = LlmConfig(model="gpt-4", provider="openai", parameters={"temperature": 0.7})
        llm3 = LlmConfig(model="gpt-4", provider="openai", parameters={"temperature": 0.5})
        llm4 = LlmConfig(model="gpt-3.5", provider="openai", parameters={"temperature": 0.7})

        assert hash(llm1) == hash(llm2)
        assert hash(llm1) != hash(llm3)
        assert hash(llm1) != hash(llm4)

        d = {llm1: "value"}
        assert d[llm2] == "value"

    def test_frozen(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        with pytest.raises(Exception):
            llm.model = "gpt-3.5"

    def test_complex_parameters_hashing(self):
        llm1 = LlmConfig(
            model="gpt-4",
            provider="openai",
            parameters={
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["\n", "END"],
                "nested": {"key": "value"}
            }
        )
        llm2 = LlmConfig(
            model="gpt-4",
            provider="openai",
            parameters={
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["\n", "END"],
                "nested": {"key": "value"}
            }
        )
        assert hash(llm1) == hash(llm2)

    def test_parameters_different_order_same_hash(self):
        llm1 = LlmConfig(
            model="gpt-4",
            provider="openai",
            parameters={"a": 1, "b": 2}
        )
        llm2 = LlmConfig(
            model="gpt-4",
            provider="openai",
            parameters={"b": 2, "a": 1}
        )
        assert hash(llm1) == hash(llm2)