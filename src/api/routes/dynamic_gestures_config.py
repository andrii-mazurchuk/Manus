"""
CRUD API for dynamic_gestures.json — the user-defined vocabulary of LSTM gestures.

Endpoints also handle hot-reloading the DynamicGestureEngine in the running pipeline
(same pattern as sequences.py → recogniser.reload_sequences()).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.gesture_event import GestureToken
from src.config.loader import load_dynamic_gestures_config, save_dynamic_gestures_config

router = APIRouter(prefix="/api/dynamic-gestures", tags=["dynamic-gestures"])

_SEQ_DIR     = Path("src/data/sequences")
_VALID_TOKENS = {t.value for t in GestureToken}
_VALID_ACTIONS = {
    "mute", "unmute", "volume_up", "volume_down",
    "next_slide", "prev_slide", "none",
}


# ── Request / response models ─────────────────────────────────────────────────

class DynamicGestureDefinition(BaseModel):
    name:           str   = Field(..., pattern=r"^[a-z0-9_]+$", min_length=1, max_length=64)
    description:    str   = ""
    action:         str   = "none"
    mirror_augment: bool  = False

    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in _VALID_ACTIONS:
            raise ValueError(f"Unknown action '{v}'. Valid: {sorted(_VALID_ACTIONS)}")
        return v


class DynamicGesturesGlobalConfig(BaseModel):
    trigger_token: str
    arm_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @classmethod
    def validate_token(cls, v: str) -> str:
        if v not in _VALID_TOKENS:
            raise ValueError(f"Unknown token '{v}'. Valid: {sorted(_VALID_TOKENS)}")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reload_engine() -> None:
    """Hot-reload DynamicGestureEngine config if the pipeline is running."""
    try:
        from src.core.event_bus import EventBus
        from src.core.dynamic_gesture_engine import DynamicGestureEngine
        bus = EventBus.get()
        for adapter in bus._subscribers:
            if isinstance(adapter, DynamicGestureEngine):
                adapter.reload_config()
                break
    except Exception:
        pass  # pipeline not running — no-op


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config():
    return load_dynamic_gestures_config()


@router.put("/config")
def put_config(payload: DynamicGesturesGlobalConfig):
    if payload.trigger_token not in _VALID_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown trigger token '{payload.trigger_token}'. "
                   f"Valid: {sorted(_VALID_TOKENS)}",
        )
    cfg = load_dynamic_gestures_config()
    cfg["trigger_token"] = payload.trigger_token
    cfg["arm_threshold"] = payload.arm_threshold
    save_dynamic_gestures_config(cfg)
    _reload_engine()
    return {"ok": True}


@router.get("/list")
def list_gestures():
    cfg = load_dynamic_gestures_config()
    return {"gestures": cfg.get("gestures", [])}


@router.post("/")
def create_gesture(payload: DynamicGestureDefinition):
    cfg = load_dynamic_gestures_config()
    gestures = cfg.setdefault("gestures", [])
    if any(g["name"] == payload.name for g in gestures):
        raise HTTPException(
            status_code=409, detail=f"Gesture '{payload.name}' already exists."
        )
    if payload.action not in _VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown action '{payload.action}'. Valid: {sorted(_VALID_ACTIONS)}",
        )
    gestures.append(payload.model_dump())
    save_dynamic_gestures_config(cfg)
    return {"ok": True, "gesture": payload.model_dump()}


@router.put("/{name}")
def update_gesture(name: str, payload: DynamicGestureDefinition):
    cfg = load_dynamic_gestures_config()
    gestures = cfg.get("gestures", [])
    for i, g in enumerate(gestures):
        if g["name"] == name:
            if payload.action not in _VALID_ACTIONS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown action '{payload.action}'. Valid: {sorted(_VALID_ACTIONS)}",
                )
            gestures[i] = payload.model_dump()
            save_dynamic_gestures_config(cfg)
            return {"ok": True}
    raise HTTPException(status_code=404, detail=f"Gesture '{name}' not found.")


@router.delete("/{name}")
def delete_gesture(name: str):
    cfg = load_dynamic_gestures_config()
    gestures = cfg.get("gestures", [])
    original_len = len(gestures)
    cfg["gestures"] = [g for g in gestures if g["name"] != name]
    if len(cfg["gestures"]) == original_len:
        raise HTTPException(status_code=404, detail=f"Gesture '{name}' not found.")
    save_dynamic_gestures_config(cfg)
    # Delete recorded sequence data for this gesture
    seq_dir = _SEQ_DIR / name
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    return {"ok": True, "deleted": name}


@router.post("/reload")
def reload_engine():
    """Hot-reload DynamicGestureEngine config in the running pipeline."""
    _reload_engine()
    return {"ok": True}
