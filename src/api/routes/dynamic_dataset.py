"""
Dataset capture API for dynamic (LSTM) gesture sequences.

Mirrors dataset.py but specialised for sequence recording:
  - Uses MediaPipe VIDEO mode for smoother temporal tracking
  - Saves one (T, 63) .npy file per recording (no augmentation at capture time)
  - Only sequence mode — start/stop recording via trigger
  - Gesture names come from dynamic_gestures.json, not GestureToken enum

Data layout:
    src/data/sequences/<gesture_name>/<timestamp_ms>.npy   shape (T, 63) float32
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.normalizer import normalize_landmarks_xyz, HAND_CONNECTIONS
from src.config.loader import load_dynamic_gestures_config

router = APIRouter(prefix="/api/dynamic-dataset", tags=["dynamic-dataset"])

_SEQ_DIR   = Path("src/data/sequences")
_MODEL_PATH = Path("src/models/hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_model() -> Path:
    import urllib.request
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _gesture_names() -> list[str]:
    cfg = load_dynamic_gestures_config()
    return [g["name"] for g in cfg.get("gestures", [])]


def _seq_stats() -> dict[str, dict]:
    stats: dict[str, dict] = {}
    if not _SEQ_DIR.exists():
        return stats
    for gesture_dir in sorted(_SEQ_DIR.iterdir()):
        if not gesture_dir.is_dir():
            continue
        files = list(gesture_dir.glob("*.npy"))
        total_frames = 0
        for f in files:
            try:
                arr = np.load(str(f), mmap_mode="r")
                total_frames += arr.shape[0]
            except Exception:
                pass
        stats[gesture_dir.name] = {
            "recordings": len(files),
            "total_frames": total_frames,
        }
    return stats


def _draw_hand(frame: np.ndarray, landmarks, color: tuple) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)


def _encode_frame(
    frame: np.ndarray,
    gesture_name: str,
    state: str,
    frames_recorded: int,
) -> bytes:
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    state_color = {
        "ready":     (200, 200, 200),
        "capturing": (50, 220, 50),
        "done":      (50, 220, 220),
    }.get(state, (200, 200, 200))

    status = f"[{state.upper()}]  gesture: {gesture_name}"
    if state == "capturing":
        status += f"  frames: {frames_recorded}"

    cv2.putText(overlay, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)

    hint = "SPACE / trigger button to start/stop recording"
    cv2.putText(overlay, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpeg.tobytes()


# ── Camera Session ────────────────────────────────────────────────────────────

class DynamicCameraSession:
    """
    Background camera session for dynamic gesture sequence recording.

    Uses MediaPipe VIDEO mode (smoother tracking than IMAGE mode) with a
    monotonically increasing timestamp counter required by the VIDEO API.

    State machine: idle → ready → capturing → done → ready
    """

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt   = threading.Event()
        self._live_jpeg: bytes | None = None

        self._camera_index: int = 0
        self._gesture_name: str = ""

        self._recording: bool = False
        self._seq_frames: list[np.ndarray] = []
        self._frames_recorded: int = 0

        self._state: str = "idle"
        self._result: dict | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active":           self.active,
                "state":            self._state,
                "gesture_name":     self._gesture_name,
                "recording":        self._recording,
                "frames_recorded":  self._frames_recorded,
                "result":           self._result,
            }

    def start(self, camera_index: int, gesture_name: str) -> None:
        self.stop()
        with self._lock:
            self._camera_index  = camera_index
            self._gesture_name  = gesture_name
            self._state         = "ready"
            self._result        = None
            self._recording     = False
            self._seq_frames    = []
            self._frames_recorded = 0
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_evt.set()
            self._thread.join(timeout=3.0)
        self._thread = None
        self._live_jpeg = None
        with self._lock:
            self._state = "idle"
            self._recording = False
            self._seq_frames = []
            self._frames_recorded = 0

    def trigger(self) -> str:
        """Toggle recording on/off. Returns action string."""
        if not self.active:
            return "no_session"
        with self._lock:
            if not self._recording:
                self._recording = True
                self._seq_frames = []
                self._frames_recorded = 0
                self._state = "capturing"
                return "recording_started"
            else:
                # Stop recording — save in background to avoid blocking
                frames = list(self._seq_frames)
                self._recording = False
                self._seq_frames = []
                gesture_name = self._gesture_name
                self._state = "done"
        # Save outside lock
        threading.Thread(
            target=self._save_sequence_npy,
            args=(frames, gesture_name),
            daemon=True,
        ).start()
        return "recording_stopped"

    def get_jpeg(self) -> bytes | None:
        return self._live_jpeg

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        detector = mp_vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self._camera_index)
        ts_ms = 0  # monotonically increasing timestamp for VIDEO mode

        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, ts_ms)
                ts_ms += 33  # ~30 fps

                with self._lock:
                    state        = self._state
                    gesture_name = self._gesture_name
                    recording    = self._recording
                    frames_rec   = self._frames_recorded

                # Draw hand if detected
                if result.hand_landmarks:
                    _draw_hand(frame, result.hand_landmarks[0],
                               color=(50, 220, 50) if recording else (200, 200, 200))

                # Accumulate frames when recording
                if recording and result.hand_landmarks:
                    frame_63 = normalize_landmarks_xyz(result.hand_landmarks[0])
                    with self._lock:
                        self._seq_frames.append(frame_63)
                        self._frames_recorded = len(self._seq_frames)
                        frames_rec = self._frames_recorded

                jpeg = _encode_frame(frame, gesture_name, state, frames_rec)
                self._live_jpeg = jpeg
        finally:
            cap.release()
            detector.close()

    def _save_sequence_npy(self, frames: list[np.ndarray], gesture_name: str) -> None:
        if not frames:
            with self._lock:
                self._result = {"error": "No frames recorded"}
                self._state = "done"
            return
        seq_dir = _SEQ_DIR / gesture_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        arr = np.stack(frames).astype(np.float32)  # (T, 63)
        ts_ms = int(time.time() * 1000)
        np.save(str(seq_dir / f"{ts_ms}.npy"), arr)

        all_files = list(seq_dir.glob("*.npy"))
        total_frames = 0
        for f in all_files:
            try:
                total_frames += np.load(str(f), mmap_mode="r").shape[0]
            except Exception:
                pass

        with self._lock:
            self._result = {
                "frames":            arr.shape[0],
                "duration_s":        round(arr.shape[0] / 30.0, 2),
                "total_recordings":  len(all_files),
                "total_frames":      total_frames,
            }
            self._state = "done"


# ── Singleton session ─────────────────────────────────────────────────────────

_session = DynamicCameraSession()


# ── Request models ────────────────────────────────────────────────────────────

class DynamicSessionStartRequest(BaseModel):
    camera:       int = 0
    gesture_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():
    return {"gestures": _seq_stats()}


@router.get("/capture/stream")
def capture_stream():
    def generate():
        while _session.active:
            jpeg = _session.get_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/session/start")
def session_start(req: DynamicSessionStartRequest):
    valid = _gesture_names()
    if req.gesture_name not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown gesture '{req.gesture_name}'. "
                   f"Define it first in Dynamic Gestures → Gestures. "
                   f"Known: {valid}",
        )
    _session.start(req.camera, req.gesture_name)
    return {"status": "started", "gesture_name": req.gesture_name}


@router.post("/session/stop")
def session_stop():
    _session.stop()
    return {"status": "stopped"}


@router.post("/session/trigger")
def session_trigger():
    action = _session.trigger()
    return {"action": action}


@router.get("/session/status")
def session_status():
    return _session.get_status()


@router.delete("/{gesture_name}")
def delete_gesture_data(gesture_name: str):
    target = _SEQ_DIR / gesture_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"No data for '{gesture_name}'")
    shutil.rmtree(target)
    return {"deleted": gesture_name}
