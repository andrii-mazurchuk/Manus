import os
import time
import requests
from dotenv import load_dotenv

from src.core.base_adapter import BaseAdapter
from src.core.event_bus import ADAPTER_DEBOUNCE_MS
from src.core.gesture_event import GestureEvent
from src.core.sequence_event import SequenceEvent

load_dotenv()

_URL          = os.getenv("WS_ADAPTER_URL", "http://localhost:8000/gesture")
_SEQUENCE_URL = os.getenv("WS_ADAPTER_SEQUENCE_URL", "http://localhost:8000/sequence")
_THRESHOLD    = float(os.getenv("WS_ADAPTER_CONFIDENCE_THRESHOLD", "0.70"))
_TIMEOUT      = float(os.getenv("WS_ADAPTER_REQUEST_TIMEOUT",      "0.1"))


class WebSocketAdapter(BaseAdapter):
    """
    Forwards GestureEvents and SequenceEvents to the REST API.

    GestureEvents → POST /gesture
    SequenceEvents → POST /sequence

    Config (loaded from .env):
        WS_ADAPTER_URL                    — gesture endpoint
        WS_ADAPTER_SEQUENCE_URL           — sequence endpoint
        WS_ADAPTER_CONFIDENCE_THRESHOLD   — ignore gesture events below this score
        WS_ADAPTER_REQUEST_TIMEOUT        — HTTP timeout in seconds
    """

    def __init__(
        self,
        url: str          = _URL,
        sequence_url: str = _SEQUENCE_URL,
        threshold: float  = _THRESHOLD,
        timeout: float    = _TIMEOUT,
    ) -> None:
        self.url          = url
        self.sequence_url = sequence_url
        self.threshold    = threshold
        self.timeout      = timeout
        self._last_fired: dict[str, float] = {}

    def on_gesture(self, event: GestureEvent) -> None:
        if event.confidence < self.threshold:
            return

        now_ms = time.monotonic() * 1000
        if now_ms - self._last_fired.get(event.gesture, 0) < ADAPTER_DEBOUNCE_MS:
            return
        self._last_fired[event.gesture] = now_ms

        try:
            requests.post(self.url, json=event.to_dict(), timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            print(f"[WebSocketAdapter] POST failed → {self.url}: {exc}")

    def on_sequence(self, event: SequenceEvent) -> None:
        try:
            requests.post(self.sequence_url, json=event.to_dict(), timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            print(f"[WebSocketAdapter] POST failed → {self.sequence_url}: {exc}")
