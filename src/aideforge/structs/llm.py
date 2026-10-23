from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .abc import BaseStruct

__all__ = ["LlmConfig"]


class LlmConfig(BaseStruct):
    # model is frozen to be hashable (for model reusability, see agent.py)
    model_config = ConfigDict(frozen=True)

    kind: Literal["llm"] = "llm"
    version: Literal[1] = 1
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
    # The token parameter is intended only for test cases or environments
    # that are guaranteed to be secure. In all other cases, store the token
    # securely outside the application and provide it through the appropriate
    # environment variable.
    # This might be later name of the API_KEY and there might be some methods
    # How to solve secure token import
    # Note: This parameter is currently not even used.
    # token: str | None = Field(
    #     defsult=None,
    #     description="Credential token to access large language model",
    # )

    @model_validator(mode="before")
    @classmethod
    def parse_model_string(cls, values: dict[str, Any]) -> dict[str, Any]:
        # support model="openai:gpt-4o" format
        if isinstance(values, dict) and values.get("provider") is None:
            model_value = values.get("model")
            if isinstance(model_value, str) and ":" in model_value:
                provider, model_name = model_value.split(":", 1)
                values["model"] = model_name
                values["provider"] = provider

        return values

    def __hash__(self) -> int:
        return hash((
            self.model,
            self.provider,
            self._hashable(self.parameters),
        ))

    @staticmethod
    def _hashable(value: Any) -> Any:
        if isinstance(value, dict):
            return frozenset(
                (key, MyModel._hashable(val))
                for key, val in value.items()
            )
        if isinstance(value, list):
            return tuple(MyModel._hashable(v) for v in value)
        return value
