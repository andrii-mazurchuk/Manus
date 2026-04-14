"""
Sequence recognition API routes.

Manages the sequences.json definition file and exposes CRUD operations so the
Studio UI can create, edit, and delete sequence definitions without touching
the file directly.

After any write operation, the running SequenceRecogniser (if registered on
the EventBus) is asked to reload its definitions so changes take effect
immediately without restarting the pipeline.
"""

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.gesture_event import GestureToken

router = APIRouter(prefix="/api/sequences", tags=["sequences"])

_SEQUENCES_PATH = Path("src/config/sequences.json")
_THRESHOLDS_PATH = Path("src/config/thresholds.json")

_VALID_TOKENS = {t.value for t in GestureToken}


# ── Pydantic models ──────────────────────────────────────────────────────────

class SequenceDefinition(BaseModel):
    name:          str              = Field(..., min_length=1, max_length=64,
                                           pattern=r"^[a-z0-9_]+$")
    pattern:       list[str]        = Field(..., min_length=2, max_length=20)
    max_gap_ms:    int | None       = Field(default=None, ge=100, le=10000)
    max_total_ms:  int | None       = Field(default=None, ge=200, le=30000)
    action:        str              = "none"


class TimingConfig(BaseModel):
    default_max_gap_ms:   int = Field(default=800,  ge=100,  le=10000)
    default_max_total_ms: int = Field(default=3000, ge=200,  le=30000)
    buffer_size:          int = Field(default=10,   ge=2,    le=50)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _read_sequences() -> list[dict]:
    if not _SEQUENCES_PATH.exists():
        return []
    with open(_SEQUENCES_PATH, encoding="utf-8") as f:
        return json.load(f).get("sequences", [])


def _write_sequences(sequences: list[dict]) -> None:
    _SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SEQUENCES_PATH, "w", encoding="utf-8") as f:
        json.dump({"sequences": sequences}, f, indent=2)


def _reload_recogniser() -> None:
    """Ask the live SequenceRecogniser to reload its definitions, if running."""
    try:
        from src.core.event_bus import EventBus
        from src.core.sequence_recogniser import SequenceRecogniser
        bus = EventBus.get()
        for adapter in bus._subscribers:
            if isinstance(adapter, SequenceRecogniser):
                adapter.reload_sequences()
                break
    except Exception:
        pass  # pipeline may not be running when Studio is used standalone


def _read_timing() -> dict:
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        sm = data.get("sequence_model") or {}
        if not isinstance(sm, dict):
            sm = {}
        return {
            "default_max_gap_ms":   sm.get("default_max_gap_ms",   800),
            "default_max_total_ms": sm.get("default_max_total_ms", 3000),
            "buffer_size":          sm.get("buffer_size",          10),
        }
    except Exception:
        return {"default_max_gap_ms": 800, "default_max_total_ms": 3000, "buffer_size": 10}


def _write_timing(cfg: dict) -> None:
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data["sequence_model"] = cfg
    with open(_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/status")
async def sequences_status():
    sequences = _read_sequences()
    return {
        "available": True,
        "sequence_count": len(sequences),
        "timing": _read_timing(),
    }


@router.get("/list")
async def list_sequences():
    return {"sequences": _read_sequences()}


@router.post("/")
async def create_sequence(body: SequenceDefinition):
    for token in body.pattern:
        if token not in _VALID_TOKENS:
            raise HTTPException(
                422,
                f"Unknown token '{token}' in pattern. "
                f"Valid tokens: {sorted(_VALID_TOKENS)}",
            )

    sequences = _read_sequences()
    if any(s["name"] == body.name for s in sequences):
        raise HTTPException(409, f"Sequence '{body.name}' already exists. Use PUT to update.")

    entry: dict = {"name": body.name, "pattern": body.pattern, "action": body.action}
    if body.max_gap_ms is not None:
        entry["max_gap_ms"] = body.max_gap_ms
    if body.max_total_ms is not None:
        entry["max_total_ms"] = body.max_total_ms

    sequences.append(entry)
    _write_sequences(sequences)
    _reload_recogniser()
    return {"ok": True, "sequence": entry}


@router.get("/timing")
async def get_timing():
    return _read_timing()


@router.put("/timing")
async def update_timing(body: TimingConfig):
    cfg = {
        "default_max_gap_ms":   body.default_max_gap_ms,
        "default_max_total_ms": body.default_max_total_ms,
        "buffer_size":          body.buffer_size,
    }
    _write_timing(cfg)
    _reload_recogniser()
    return {"ok": True, "timing": cfg}


@router.put("/{name}")
async def update_sequence(name: str, body: SequenceDefinition):
    if body.name != name:
        raise HTTPException(422, "Name in body must match URL parameter.")

    for token in body.pattern:
        if token not in _VALID_TOKENS:
            raise HTTPException(
                422,
                f"Unknown token '{token}' in pattern. "
                f"Valid tokens: {sorted(_VALID_TOKENS)}",
            )

    sequences = _read_sequences()
    idx = next((i for i, s in enumerate(sequences) if s["name"] == name), None)
    if idx is None:
        raise HTTPException(404, f"Sequence '{name}' not found.")

    entry: dict = {"name": body.name, "pattern": body.pattern, "action": body.action}
    if body.max_gap_ms is not None:
        entry["max_gap_ms"] = body.max_gap_ms
    if body.max_total_ms is not None:
        entry["max_total_ms"] = body.max_total_ms

    sequences[idx] = entry
    _write_sequences(sequences)
    _reload_recogniser()
    return {"ok": True, "sequence": entry}


@router.delete("/{name}")
async def delete_sequence(name: str):
    sequences = _read_sequences()
    original_len = len(sequences)
    sequences = [s for s in sequences if s["name"] != name]
    if len(sequences) == original_len:
        raise HTTPException(404, f"Sequence '{name}' not found.")
    _write_sequences(sequences)
    _reload_recogniser()
    return {"ok": True, "deleted": name}
