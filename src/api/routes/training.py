import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.training.trainer import DEFAULT_CSV, DEFAULT_MODEL, run_training, validate_dataset

router = APIRouter(prefix="/api/train", tags=["training"])

# ── Paths ─────────────────────────────────────────────────────────────────────
_CSV_PATH   = DEFAULT_CSV
_MODEL_PATH = DEFAULT_MODEL
_HISTORY_MAX = 5

# ── Module-level state (singleton — persists for the lifetime of the server) ──
_state: dict = {
    "status":       "idle",   # "idle" | "running" | "done" | "error"
    "progress":     "",
    "result":       None,     # metrics dict after a successful run
    "last_trained": None,     # ISO-8601 string
    "error":        None,     # error message when status == "error"
}
_lock    = threading.Lock()
_history: list[dict] = []


# ── Background worker ─────────────────────────────────────────────────────────

def _do_train() -> None:
    def _progress(msg: str) -> None:
        with _lock:
            _state["progress"] = msg

    with _lock:
        _state.update(status="running", progress="Starting…", error=None, result=None)

    try:
        result = run_training(_CSV_PATH, _MODEL_PATH, progress_cb=_progress)
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
async def start_training():
    """
    Trigger background model training.

    Returns {"status": "started"} or {"status": "already_running"}.
    Raises 400 if the dataset is missing or too small to train on.
    """
    with _lock:
        if _state["status"] == "running":
            return {"status": "already_running"}

    # Pre-flight: fast validation before spawning the thread.
    # validate_dataset() reads only the label column — negligible I/O.
    try:
        validate_dataset(_CSV_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    threading.Thread(target=_do_train, daemon=True).start()
    return {"status": "started"}


@router.get("/status")
async def get_training_status():
    """
    Return the current training state.

    Response shape:
        {
            "status":       "idle" | "running" | "done" | "error",
            "progress":     "<current step message>",
            "result":       null | { model, accuracy, cv_score, per_class },
            "last_trained": null | "<ISO-8601 timestamp>",
            "error":        null | "<error message>",
        }
    """
    with _lock:
        return deepcopy(_state)


@router.get("/history")
async def get_training_history():
    """Return the last 5 training runs."""
    with _lock:
        return {"history": list(_history)}
