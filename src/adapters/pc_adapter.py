import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pynput.keyboard import Controller, Key

from src.core.base_adapter import BaseAdapter
from src.core.event_bus import ADAPTER_DEBOUNCE_MS
from src.core.gesture_event import GestureEvent, GestureToken

load_dotenv()

_THRESHOLD = float(os.getenv("PC_ADAPTER_CONFIDENCE_THRESHOLD", "0.70"))

_CONFIG_PATH = Path(__file__).parents[1] / "config" / "gesture_actions.json"

_DEFAULTS: dict[str, str] = {
    "STOP":    "mute",
    "PLAY":    "unmute",
    "UP":      "volume_up",
    "DOWN":    "volume_down",
    "CONFIRM": "next_slide",
    "CANCEL":  "prev_slide",
    "MODE":    "none",
    "CUSTOM":  "none",
}


class PCAdapter(BaseAdapter):
    def __init__(self, threshold: float = _THRESHOLD) -> None:
        self.threshold = threshold
        self._last_fired: dict[str, float] = {}
        self._kb = Controller()
        self._token_actions = self._load_actions()
        self._action_map = self._build_action_map()

    def _load_actions(self) -> dict[str, str]:
        try:
            data = json.loads(_CONFIG_PATH.read_text())
            return data["static_actions"]
        except Exception:
            return dict(_DEFAULTS)

    def _press(self, key) -> None:
        self._kb.press(key)
        self._kb.release(key)

    def _build_action_map(self) -> dict[str, object]:
        press = self._press
        return {
            "mute":        lambda: press(Key.media_volume_mute),
            "unmute":      lambda: press(Key.media_volume_mute),
            "volume_up":   lambda: press(Key.media_volume_up),
            "volume_down": lambda: press(Key.media_volume_down),
            "next_slide":  lambda: press(Key.right),
            "prev_slide":  lambda: press(Key.left),
            "none":        lambda: None,
        }

    def on_gesture(self, event: GestureEvent) -> None:
        if event.confidence < self.threshold:
            return
        token = event.gesture
        now = time.monotonic() * 1000
        if now - self._last_fired.get(token, 0) < ADAPTER_DEBOUNCE_MS:
            return
        self._last_fired[token] = now
        self._execute(token)

    def _execute(self, token: GestureToken) -> None:
        try:
            action_name = self._token_actions.get(token.value, "none")
            fn = self._action_map.get(action_name)
            if fn:
                fn()
        except Exception as exc:
            print(f"[PCAdapter] key action failed for {token}: {exc}")
