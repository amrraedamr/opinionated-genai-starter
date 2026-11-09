from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsRealtimeConfig(Protocol):
    """Satisfied by `RealtimeSettingsMixin`."""

    realtime_provider: str
    realtime_model: str
    realtime_voice: str
    realtime_instructions: str
    realtime_audio_enabled: bool
