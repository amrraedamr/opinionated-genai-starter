from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsPromptConfig(Protocol):
    """Satisfied by `PromptSettingsMixin`."""

    service_name: str
    load_prompts_from_db: bool
