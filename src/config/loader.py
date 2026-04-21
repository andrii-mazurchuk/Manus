"""
Config file read/write helpers for Manus.

Centralises JSON I/O for dynamic_gestures.json so neither main.py nor API
routes duplicate file-reading boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent

_DEFAULT_DYNAMIC: dict = {
    "trigger_token": "UP",
    "arm_threshold": 0.75,
    "gestures": [],
}


def load_dynamic_gestures_config() -> dict:
    path = _CONFIG_DIR / "dynamic_gestures.json"
    if not path.exists():
        return dict(_DEFAULT_DYNAMIC)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Back-fill any missing top-level keys so callers never KeyError
    for k, v in _DEFAULT_DYNAMIC.items():
        data.setdefault(k, v)
    return data


def save_dynamic_gestures_config(data: dict) -> None:
    path = _CONFIG_DIR / "dynamic_gestures.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
