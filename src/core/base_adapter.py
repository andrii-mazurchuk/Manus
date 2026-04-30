"""
BaseAdapter — abstract base class for all Manus output adapters.

Every adapter (PC control, WebSocket, MQTT, etc.) subclasses this and
implements one method: on_gesture(). The event bus calls it for every
GestureEvent that passes the confidence threshold.

Implementing a new adapter:
    from src.core.base_adapter import BaseAdapter
    from src.core.gesture_event import GestureEvent

    class MyAdapter(BaseAdapter):
        def on_gesture(self, event: GestureEvent) -> None:
            print(f"Got {event.gesture.value} with {event.confidence:.0%}")
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .gesture_event import GestureEvent

if TYPE_CHECKING:
    from .sequence_event import SequenceEvent


class BaseAdapter(ABC):
    @abstractmethod
    def on_gesture(self, event: GestureEvent) -> None:
        """Called by the event bus for every emitted GestureEvent."""
        ...

    def on_sequence(self, event: "SequenceEvent") -> None:
        """Called by the event bus when a gesture sequence is matched.

        Default implementation is a no-op. Override in adapters that should
        react to sequence events (e.g. PCAdapter, WebSocketAdapter).
        """
        pass
