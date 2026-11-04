from pydantic import BaseModel


class LLMSettingsMixin(BaseModel):
    """Generation knobs that hold whichever provider is behind them.

    Compose it with the provider mixin the service actually configures, the
    way `SQLSettingsMixin` pairs with `PostgresSettingsMixin`.
    """

    capture_raw_traces: bool = False
    """Capture raw prompts and responses in the debug trace."""
