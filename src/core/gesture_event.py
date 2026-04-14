"""
Single source of truth for gesture vocabulary and the GestureEvent contract.

Both the ML pipeline (Person A) and the REST/WS layer (Person B) import from
here. Changing a token name in this file is the *only* change needed across
the whole system.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field


class GestureToken(str, Enum):
    """
    Canonical gesture labels emitted by the event bus.

    Inheriting from `str` means every token IS a plain string — serialises to
    JSON as "STOP", not as {"value": "STOP"} — while still being validated by
    FastAPI / Pydantic automatically.

    Token   | Gesture               | Default PC action
    --------|-----------------------|---------------------------
    STOP    | Fist                  | Mute / pause
    PLAY    | Open palm             | Unmute / resume
    UP      | Index pointing up     | Volume up / scroll up
    DOWN    | Index pointing down   | Volume down / scroll down
    CONFIRM | Thumbs up             | Next slide / confirm
    CANCEL  | Thumbs down           | Previous slide / cancel
    MODE    | Peace / V             | Switch adapter mode
    CUSTOM  | Shaka                 | User-defined
    SNAP    | Finger snap           | Single-hand; user-defined
    CLAP    | Two hands together    | Two-hand; user-defined
    """

    STOP    = "STOP"
    PLAY    = "PLAY"
    UP      = "UP"
    DOWN    = "DOWN"
    CONFIRM = "CONFIRM"
    CANCEL  = "CANCEL"
    MODE    = "MODE"
    CUSTOM  = "CUSTOM"
    SNAP    = "SNAP"
    CLAP    = "CLAP"


@dataclass
class GestureEvent:
    """
    The single object flowing through the entire Manus pipeline.

    Used by:
      - the ML classifier  → creates events
      - the event bus      → routes events
      - every adapter      → receives events via on_gesture()
      - the REST/WS API    → serialises events to JSON

    JSON representation (REST + WebSocket):
        {"gesture": "STOP", "confidence": 0.94, "label": 0, "timestamp": 1714123456.789}
    """

    gesture:    GestureToken
    confidence: float
    label:      int
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "gesture":    self.gesture.value,
            "confidence": round(self.confidence, 4),
            "label":      self.label,
            "timestamp":  self.timestamp,
        }
