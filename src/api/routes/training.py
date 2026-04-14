import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.training.trainer import (
    DEFAULT_CSV,
    DEFAULT_MODEL,
    DEFAULT_TWO_HAND_CSV,
    DEFAULT_TWO_HAND_MODEL,
    run_training,
    validate_dataset,
)

router = APIRouter(prefix="/api/train", tags=["training"])

_HISTORY_MAX = 5

# ── Per-model-type state (each has its own idle/running/done/error lifecycle) ─
def _empty_state() -> dict:
    return {
        "status":       "idle",
        "progress":     "",
        "result":       None,
        "last_trained": None,
        "error":        None,
    }

_states: dict[str, dict] = {
    "single":    _empty_state(),
    "two_hand":  _empty_state(),
}
_lock    = threading.Lock()
_history: list[dict] = []


# ── Background worker ─────────────────────────────────────────────────────────

def _do_train(model_type: str) -> None:
    csv_path   = DEFAULT_TWO_HAND_CSV   if model_type == "two_hand" else DEFAULT_CSV
    model_path = DEFAULT_TWO_HAND_MODEL if model_type == "two_hand" else DEFAULT_MODEL

    def _progress(msg: str) -> None:
        with _lock:
            _states[model_type]["progress"] = msg

    with _lock:
        _states[model_type].update(status="running", progress="Starting…", error=None, result=None)

    try:
        result = run_training(csv_path, model_path, progress_cb=_progress)
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            _states[model_type].update(status="done", result=result, last_trained=ts, progress="")
            _history.insert(0, {**result, "model_type": model_type, "timestamp": ts})
            del _history[_HISTORY_MAX:]
    except Exception as exc:
        with _lock:
            _states[model_type].update(status="error", error=str(exc), progress="")


# ── Endpoints ─────────────────────────────────────────────────────────────────

class TrainStartRequest(BaseModel):
    model_type: Literal["single", "two_hand"] = "single"


@router.post("/start")
async def start_training(body: TrainStartRequest = TrainStartRequest()):
    """
    Trigger background model training.

    Pass {"model_type": "two_hand"} to train the two-hand classifier.
    Returns {"status": "started"} or {"status": "already_running"}.
    Raises 400 if the dataset is missing or too small to train on.
    """
    csv_path = DEFAULT_TWO_HAND_CSV if body.model_type == "two_hand" else DEFAULT_CSV

    with _lock:
        if _states[body.model_type]["status"] == "running":
            return {"status": "already_running"}

    try:
        validate_dataset(csv_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    threading.Thread(target=_do_train, args=(body.model_type,), daemon=True).start()
    return {"status": "started", "model_type": body.model_type}


@router.get("/status")
async def get_training_status(model_type: Literal["single", "two_hand"] = "single"):
    """
    Return the current training state for the given model type.

    Pass ?model_type=two_hand to poll two-hand training progress.
    """
    with _lock:
        return deepcopy(_states[model_type])


@router.get("/history")
async def get_training_history():
    """Return the last 5 training runs across all model types."""
    with _lock:
        return {"history": list(_history)}
