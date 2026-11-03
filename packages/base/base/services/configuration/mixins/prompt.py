from pydantic import BaseModel


class PromptSettingsMixin(BaseModel):
    load_prompts_from_db: bool = False
    """Apply stored prompt overrides at startup and follow reload events."""
