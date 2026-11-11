from pydantic import Field

from base.models.view_models import BaseModel, JSONObject


class NotificationRequest(BaseModel):
    payload: JSONObject = Field(default_factory=dict)


class NotificationResponse(BaseModel):
    seq: int
    """Where it sits in the caller's inbox; a device resumes from below it."""
