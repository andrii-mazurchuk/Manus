from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT        = Path(__file__).parent.parent
_GESTURE_LABELS_PATH = _PROJECT_ROOT / "src" / "config" / "gesture_labels.json"
_SEQUENCES_PATH      = _PROJECT_ROOT / "src" / "config" / "sequences.json"
_THRESHOLDS_PATH     = _PROJECT_ROOT / "src" / "config" / "thresholds.json"


class ConfigManager:
    _instance: ConfigManager | None = None

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Gesture labels ─────────────────────────────────────────────────────

    def load_gesture_labels(self) -> list[dict]:
        """Returns list of {name: str, threshold: float}."""
        try:
            with open(_GESTURE_LABELS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("gestures", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_gesture_labels(self, gestures: list[dict]) -> None:
        _GESTURE_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_GESTURE_LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump({"gestures": gestures}, f, indent=2)

    def gesture_names(self) -> list[str]:
        return [g["name"] for g in self.load_gesture_labels()]

    # ── Sequences ──────────────────────────────────────────────────────────

    def load_sequences(self) -> dict:
        try:
            with open(_SEQUENCES_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"sequences": []}

    def save_sequences(self, data: dict) -> None:
        _SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SEQUENCES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Thresholds ─────────────────────────────────────────────────────────

    def load_thresholds(self) -> dict:
        try:
            with open(_THRESHOLDS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_thresholds(self, data: dict) -> None:
        _THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Reload all ─────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of error strings. Empty list = all good."""
        errors: list[str] = []
        for path, label in [
            (_GESTURE_LABELS_PATH, "gesture_labels.json"),
            (_SEQUENCES_PATH,      "sequences.json"),
            (_THRESHOLDS_PATH,     "thresholds.json"),
        ]:
            if not path.exists():
                errors.append(f"Missing: {label}")
            else:
                try:
                    with open(path, encoding="utf-8") as f:
                        json.load(f)
                except json.JSONDecodeError as exc:
                    errors.append(f"Invalid JSON in {label}: {exc}")
        return errors
