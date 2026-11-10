from pydantic import BaseModel


class RealtimeSettingsMixin(BaseModel):
    """Which model holds an open conversation, and how it answers."""

    realtime_provider: str = "openai"
    """Separate from `llm_provider`; must be a provider with a realtime API."""

    realtime_model: str = "gpt-realtime"
    realtime_voice: str = ""
    """Provider-specific; empty leaves the provider's own default."""
    realtime_instructions: str = ""
    """What the model is told before anyone speaks. Empty leaves it default."""

    realtime_audio_enabled: bool = True
    """Off makes the conversation text-only."""
