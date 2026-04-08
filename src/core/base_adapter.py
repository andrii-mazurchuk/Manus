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

from .gesture_event import GestureEvent


class BaseAdapter(ABC):
    @abstractmethod
    def on_gesture(self, event: GestureEvent) -> None:
        """Called by the event bus for every emitted GestureEvent."""
        ...
