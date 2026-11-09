from pydantic import BaseModel


class RealtimeSettingsMixin(BaseModel):
    """Which model holds an open conversation, and how it answers."""

    realtime_provider: str = "openai"
    """Named separately from `llm_provider`: a service can answer requests
    with one provider and hold conversations with another, and not every
    provider has a realtime API at all."""

    realtime_model: str = "gpt-realtime"
    realtime_voice: str = "alloy"
    realtime_instructions: str = ""
    """What the model is told before anyone speaks. Empty leaves it default."""

    realtime_audio_enabled: bool = True
    """Off makes it a typed conversation, which is cheaper and sometimes all
    a service wants."""
