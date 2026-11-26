from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .abc import BaseStruct

__all__ = ["LlmConfig"]


class LlmConfig(BaseStruct):
    """Configuration for a Large Language Model.

    .. note::
       This model is frozen to be hashable (for the sake of model reusability).
    """

    # See model reusability in agent/runtime.py#_get_model_instance()
    model_config = ConfigDict(frozen=True)

    kind: Literal["llm"] = Field(
        default="llm",
        description='The type identifier, always "llm".',
    )
    version: Literal[1] = Field(
        default=1,
        description="Schema version.",
    )
    model: str = Field(
        ...,
        description=(
            "Large language model's name. 'provider:modelname' format is "
            "supported."
        ),
    )
    provider: str = Field(
        ..., description="Large language model's provider's name"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Large language model's special parameters (like temperature).",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_model_string(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Support ``<provider>:<model>`` format in configuration.

        The method checks whether the provider is missing. If so, and the
        model string contains one, it separates the provider from the model
        and stores both correctly in the model.
        """
        # support model="openai:gpt-4o" format
        if isinstance(values, dict) and values.get("provider") is None:
            model_value = values.get("model")
            if isinstance(model_value, str) and ":" in model_value:
                provider, model_name = model_value.split(":", 1)
                values["model"] = model_name
                values["provider"] = provider

        return values

    def __hash__(self) -> int:
        return hash(
            (
                self.model,
                self.provider,
                self._hashable(self.parameters),
            )
        )

    @staticmethod
    def _hashable(value: Any) -> Any:
        if isinstance(value, dict):
            return frozenset(
                (key, LlmConfig._hashable(val)) for key, val in value.items()
            )
        if isinstance(value, list):
            return tuple(LlmConfig._hashable(v) for v in value)
        return value
