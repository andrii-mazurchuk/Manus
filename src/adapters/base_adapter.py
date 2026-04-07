from abc import ABC, abstractmethod
from src.core.gesture_event import GestureEvent



class BaseAdapter(ABC):
    @abstractmethod
    def on_gesture(self, event: GestureEvent) -> None:
        """Called by the event bus for every emitted GestureEvent."""
        pass