"""
SequenceEvent — the object emitted when a gesture sequence pattern is matched.

SequenceEvent is separate from GestureEvent: it represents a temporal pattern
of static gesture tokens rather than a single-frame classification. Adapters
that care about sequences implement on_sequence(); all others receive a default
no-op and are unaffected.

JSON representation (WebSocket + REST):
    {
        "type":       "sequence",
        "name":       "double_clap",
        "tokens":     ["CLAP", "CLAP"],
        "confidence": 0.87,
        "timestamp":  1714123456.789,
        "duration":   0.42
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .gesture_event import GestureToken


@dataclass
class SequenceEvent:
    """
    Emitted by SequenceRecogniser when a defined token pattern is matched.

    Fields:
        name        — sequence name as defined in sequences.json (e.g. "double_clap")
        tokens      — the ordered list of GestureTokens that triggered the match
        confidence  — mean confidence of the matched GestureEvents (0.0 – 1.0)
        timestamp   — Unix timestamp of the final token in the match
        duration    — seconds between first and last matched token
    """

    name:       str
    tokens:     list[GestureToken]
    confidence: float
    timestamp:  float = field(default_factory=time.time)
    duration:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "type":       "sequence",
            "name":       self.name,
            "tokens":     [t.value for t in self.tokens],
            "confidence": round(self.confidence, 4),
            "timestamp":  self.timestamp,
            "duration":   round(self.duration, 4),
        }
