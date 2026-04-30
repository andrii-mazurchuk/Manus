"""
LSTM sequence model training API.

Mirrors training.py pattern exactly: background thread, progress polling,
history list. Only one model type so no model_type parameter needed.
"""

import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.training.sequence_trainer import (
    run_sequence_training,
    validate_sequence_dataset,
)

router = APIRouter(prefix="/api/dynamic-train", tags=["dynamic-training"])

_HISTORY_MAX = 5
_SEQ_DATA_DIR = Path("src/data/sequences")

_lock    = threading.Lock()
_state: dict = {
    "status":       "idle",
    "progress":     "",
    "result":       None,
    "last_trained": None,
    "error":        None,
}
_history: list[dict] = []


# ── Background worker ─────────────────────────────────────────────────────────

def _do_train() -> None:
    def _progress(msg: str) -> None:
        with _lock:
            _state["progress"] = msg

    with _lock:
        _state.update(status="running", progress="Starting…", error=None, result=None)

    try:
        result = run_sequence_training(progress_cb=_progress)
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            _state.update(status="done", result=result, last_trained=ts, progress="")
            _history.insert(0, {**result, "timestamp": ts})
            del _history[_HISTORY_MAX:]
    except Exception as exc:
        with _lock:
            _state.update(status="error", error=str(exc), progress="")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
def start_training():
    """
    Validate sequence data and launch LSTM training in background.
    Returns {"status": "started"} or {"status": "already_running"}.
    Raises 400 if data is missing or insufficient.
    """
    with _lock:
        if _state["status"] == "running":
            return {"status": "already_running"}

    try:
        validate_sequence_dataset(_SEQ_DATA_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    threading.Thread(target=_do_train, daemon=True).start()
    return {"status": "started"}


@router.get("/status")
def get_training_status():
    """Return current LSTM training state."""
    with _lock:
        return deepcopy(_state)


@router.get("/history")
def get_training_history():
    """Return last 5 LSTM training runs."""
    with _lock:
        return {"history": list(_history)}
