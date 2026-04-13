import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from src.core.gesture_event import GestureToken

router = APIRouter(prefix="/api/config", tags=["config"])

_CONFIG_DIR = Path(__file__).parents[2] / "config"
_ACTIONS_PATH = _CONFIG_DIR / "gesture_actions.json"
_THRESHOLDS_PATH = _CONFIG_DIR / "thresholds.json"

_VALID_TOKENS = {t.value for t in GestureToken}
_VALID_ACTIONS = {"mute", "unmute", "volume_up", "volume_down", "next_slide", "prev_slide", "none"}


class ActionsPayload(BaseModel):
    static_actions: dict[str, str]
    sequence_actions: dict = {}

    @field_validator("static_actions")
    @classmethod
    def validate_static(cls, v: dict[str, str]) -> dict[str, str]:
        for token, action in v.items():
            if token not in _VALID_TOKENS:
                raise ValueError(f"Unknown gesture token: {token!r}")
            if action not in _VALID_ACTIONS:
                raise ValueError(f"Unknown action: {action!r}. Valid: {sorted(_VALID_ACTIONS)}")
        return v


class ThresholdsPayload(BaseModel):
    pc_adapter: float
    websocket_adapter: float
    sequence_model: float | None = None

    @field_validator("pc_adapter", "websocket_adapter")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt config file: {exc}")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


@router.get("/actions")
def get_actions():
    return _read_json(_ACTIONS_PATH)


@router.put("/actions")
def put_actions(payload: ActionsPayload):
    _write_json(_ACTIONS_PATH, payload.model_dump())
    return {"ok": True}


@router.get("/thresholds")
def get_thresholds():
    return _read_json(_THRESHOLDS_PATH)


@router.put("/thresholds")
def put_thresholds(payload: ThresholdsPayload):
    _write_json(_THRESHOLDS_PATH, payload.model_dump())
    return {"ok": True}
