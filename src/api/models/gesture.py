"""
Pydantic request/response model for the REST API.

Wraps core.GestureEvent so FastAPI can validate and serialise it.
The `gesture` field uses GestureToken, which means FastAPI will:
  - accept only the 8 canonical token strings
  - return HTTP 422 with a clear message for anything else
"""

from pydantic import BaseModel
from src.core.gesture_event import GestureToken


class GestureEvent(BaseModel):
    gesture:   GestureToken
    label:     int
    timestamp: float

    model_config = {"use_enum_values": True}
