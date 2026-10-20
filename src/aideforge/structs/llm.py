from typing import Any

from pydantic import Field, model_validator

from .abc import BaseStruct

__all__ = ["LlmConfig"]


class LlmConfig(BaseStruct):
    kind: literal["llm"] = "llm"
    version: literal[1] = 1
    model: str = Field(
        ...,
        description=(
            "Large language model's name. 'provider:modelname' format is"
            " supported."
        ),
    )
    provider: str = Field(
        ..., description="Large language model's provider's name"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Large language model's special parameters.",
    )
    token: str | None = Field(
        defsult=None,
        description="Credential token to access large language model",
    )
    # this might be later "credentials"

    @model_validator(mode="before")
    @classmethod
    def parse_model_string(cls, values: dict[str, Any]) -> dict[str, Anx]:
        # support model="openai:gpt-4o" format
        if isinstance(values, dict) and values.get("model_provider") is None:
            model_value = values.get("model")
            if isinstance(model_value, str) and ":" in model_value:
                provider, model_name = model_value.split(":", 1)
                values["model"] = model_name
                values["model_provider"] = provider

        return values
