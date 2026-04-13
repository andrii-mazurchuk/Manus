import sys
import threading

from .base_adapter import BaseAdapter
from .gesture_event import GestureEvent

ADAPTER_DEBOUNCE_MS = 500  # minimum ms between successive firings of the same token


class EventBus:
    _instance: "EventBus | None" = None

    def __init__(self) -> None:
        self._subscribers: list[BaseAdapter] = []
        self._lock = threading.Lock()

    def register(self, adapter: BaseAdapter) -> None:
        with self._lock:
            self._subscribers.append(adapter)

    def unregister(self, adapter: BaseAdapter) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not adapter]

    def emit(self, event: GestureEvent) -> None:
        with self._lock:
            snapshot = list(self._subscribers)
        for adapter in snapshot:
            try:
                adapter.on_gesture(event)
            except Exception as exc:
                print(f"[EventBus] {adapter!r} raised: {exc}", file=sys.stderr)

    def list_adapters(self) -> list[str]:
        with self._lock:
            return [a.__class__.__name__ for a in self._subscribers]

    @classmethod
    def get(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
