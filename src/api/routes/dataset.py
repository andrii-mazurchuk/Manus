import asyncio
import csv
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.gesture_event import GestureToken
from src.core.normalizer import normalize_landmarks, normalize_two_hand_landmarks, HAND_CONNECTIONS

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# ── Paths (relative to repo root — where uvicorn is launched) ────────────────
_CSV_PATH          = Path("src/data/gestures.csv")
_NPY_DIR           = Path("src/data/gestures")
_CSV_TWO_HAND_PATH = Path("src/data/gestures_two_hand.csv")
_NPY_TWO_HAND_DIR  = Path("src/data/gestures_two_hand")
_MODEL_PATH        = Path("src/models/hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Single-hand: label + 42 floats (x0,y0 … x20,y20)
HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

# Two-hand: label + 84 floats (primary hand _p, secondary hand _s)
TWO_HAND_HEADER = (
    ["label"]
    + [f"{axis}{i}_p" for i in range(21) for axis in ("x", "y")]
    + [f"{axis}{i}_s" for i in range(21) for axis in ("x", "y")]
)

_VALID_LABELS = {t.value for t in GestureToken}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_model() -> Path:
    import urllib.request
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _csv_label_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not _CSV_PATH.exists():
        return counts
    with open(_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def _csv_two_hand_label_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not _CSV_TWO_HAND_PATH.exists():
        return counts
    with open(_CSV_TWO_HAND_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def _probe_cameras(max_index: int = 4) -> list[dict]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append({"index": i, "label": f"Camera {i}"})
        else:
            cap.release()
    return found


def _find_dataset_root(base: Path) -> Path | None:
    def _has_labels(d: Path) -> bool:
        return any(
            sub.is_dir() and sub.name.upper() in _VALID_LABELS
            for sub in d.iterdir()
        )
    if _has_labels(base):
        return base
    for child in base.iterdir():
        if child.is_dir() and _has_labels(child):
            return child
    return None


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_hand(frame: np.ndarray, landmarks, color: tuple) -> None:
    """Draw skeleton lines and landmark dots for one hand onto frame (in-place)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)


def _encode_annotated(
    frame: np.ndarray,
    target_label: str,
    prediction: str | None,
    confidence: float | None,
    status_line: str,
) -> bytes:
    """Overlay prediction and status text onto frame, return JPEG bytes."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Top-left: prediction + confidence
    if prediction is not None:
        matched = prediction == target_label
        pred_color = (50, 220, 50) if matched else (30, 140, 255)
        pred_text = f"{prediction}  {confidence:.0%}"
        cv2.putText(overlay, pred_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, pred_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "No hand", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)

    # Top-right: target label
    target_text = f"Target: {target_label}"
    (tw, _), _ = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(overlay, target_text, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, target_text, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    # Bottom-left: status line
    if status_line:
        cv2.putText(overlay, status_line, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, status_line, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 2, cv2.LINE_AA)

    _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpeg.tobytes()


# ── Camera Session ────────────────────────────────────────────────────────────

class CameraSession:
    """
    Persistent camera session for the Dataset UI.

    A single background thread owns the camera and MediaPipe detector.
    It continuously reads frames, runs landmark detection, draws annotations,
    and writes JPEG frames to an in-memory buffer that the MJPEG stream
    endpoint reads from — so the preview never stops.

    Capture is trigger-based, not time-based:
        snapshot  — one trigger → one clean frame → AugmentationEngine → N rows
        sequence  — trigger 1 → record frames → trigger 2 → stop → save raw rows
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._live_jpeg: bytes | None = None

        # Config (written by start(); read by background thread)
        self._camera_index: int = 0
        self._label: str = ""
        self._capture_type: str = "snapshot"
        self._mode: str = "single"
        self._samples_per_trigger: int = 50

        # Snapshot trigger
        self._trigger_evt = threading.Event()

        # Sequence recording
        self._recording: bool = False
        self._seq_rows: list[np.ndarray] = []

        # State machine  idle → ready → capturing → done → ready …
        self._state: str = "idle"
        self._result: dict | None = None
        self._flash_until: float = 0.0   # monotonic timestamp; shows "SAVED!" flash

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, camera_index: int, label: str, capture_type: str,
              mode: str, samples_per_trigger: int) -> None:
        self.stop()
        with self._lock:
            self._camera_index = camera_index
            self._label = label
            self._capture_type = capture_type
            self._mode = mode
            self._samples_per_trigger = samples_per_trigger
            self._state = "ready"
            self._result = None
            self._recording = False
            self._seq_rows = []
        self._stop_evt.clear()
        self._trigger_evt.clear()
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
            self._seq_rows = []

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label
            # Reset done state so the next trigger starts fresh
            if self._state == "done":
                self._state = "ready"
                self._result = None

    def trigger(self) -> str:
        """Fire the capture trigger. Returns the action taken (non-blocking)."""
        if not self.active:
            return "no_session"

        with self._lock:
            ct = self._capture_type
            state = self._state
            recording = self._recording

        if ct == "snapshot":
            if state == "capturing":
                return "busy"          # previous save still in progress
            self._trigger_evt.set()
            return "triggered"

        # sequence mode
        if not recording:
            with self._lock:
                self._recording = True
                self._seq_rows = []
                self._state = "capturing"
            return "recording_started"
        else:
            with self._lock:
                self._recording = False
                rows = list(self._seq_rows)
                self._seq_rows = []
                self._state = "capturing"
            # Save in background so the HTTP response returns immediately
            threading.Thread(target=self._save_sequence, args=(rows,), daemon=True).start()
            return "recording_stopped"

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active": self.active,
                "state": self._state,
                "result": self._result,
                "label": self._label,
                "capture_type": self._capture_type,
                "mode": self._mode,
                "recording": self._recording,
            }

    def get_jpeg(self) -> bytes | None:
        return self._live_jpeg

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Load classifiers (optional — no crash if models missing)
        clf = None
        try:
            from src.core.classifier import GestureClassifier
            clf = GestureClassifier()
        except Exception:
            pass

        two_clf = None
        try:
            from src.core.two_hand_classifier import TwoHandGestureClassifier
            two_clf = TwoHandGestureClassifier()
        except Exception:
            pass

        try:
            model_path = _ensure_model()
        except Exception:
            with self._lock:
                self._state = "idle"
            return

        cap = None
        for _ in range(6):
            cap = cv2.VideoCapture(self._camera_index)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.3)
        if cap is None or not cap.isOpened():
            with self._lock:
                self._state = "idle"
            return

        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,  # always 2 — avoids restarts when mode changes
            min_hand_detection_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )

        try:
            with mp_vision.HandLandmarker.create_from_options(options) as detector:
                while not self._stop_evt.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.01)
                        continue

                    # Snapshot config under lock (cheap copy)
                    with self._lock:
                        label       = self._label
                        mode        = self._mode
                        state       = self._state
                        recording   = self._recording
                        spt         = self._samples_per_trigger

                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_img)

                    prediction: str | None  = None
                    confidence: float | None = None
                    flat: np.ndarray | None  = None

                    if result.hand_landmarks:
                        if mode == "two_hand":
                            right_hand = left_hand = None
                            for i, lms in enumerate(result.hand_landmarks):
                                side = result.handedness[i][0].category_name
                                if side == "Right":
                                    right_hand = lms
                                else:
                                    left_hand = lms
                            primary   = right_hand or left_hand
                            secondary = left_hand if right_hand else None

                            if two_clf:
                                try:
                                    prediction, confidence = two_clf.predict(primary, secondary)
                                except Exception:
                                    pass

                            skel_color = (50, 220, 50) if (not prediction or prediction == label) else (30, 140, 255)
                            _draw_hand(frame, primary, skel_color)
                            if secondary:
                                _draw_hand(frame, secondary, (50, 180, 255))

                            flat = normalize_two_hand_landmarks(primary, secondary)
                        else:
                            lms = result.hand_landmarks[0]
                            flat = normalize_landmarks(lms)

                            if clf:
                                try:
                                    prediction, confidence = clf.predict(flat)
                                except Exception:
                                    pass

                            skel_color = (50, 220, 50) if (not prediction or prediction == label) else (30, 140, 255)
                            _draw_hand(frame, lms, skel_color)

                        # Accumulate for sequence recording
                        if recording and flat is not None:
                            with self._lock:
                                self._seq_rows.append(flat.copy())

                    # Recording indicator (red dot + "REC" text)
                    if recording:
                        h, w = frame.shape[:2]
                        cv2.circle(frame, (w - 22, 22), 9, (30, 30, 200), -1, cv2.LINE_AA)
                        cv2.putText(frame, "REC", (w - 60, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 200), 2, cv2.LINE_AA)

                    # "SAVED!" flash after snapshot trigger
                    now = time.monotonic()
                    if now < self._flash_until:
                        h, w = frame.shape[:2]
                        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 6)
                        cv2.putText(frame, "SAVED!", (w // 2 - 55, h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
                        cv2.putText(frame, "SAVED!", (w // 2 - 55, h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

                    # Handle snapshot trigger
                    if self._trigger_evt.is_set():
                        self._trigger_evt.clear()
                        if flat is not None and state != "capturing":
                            self._flash_until = time.monotonic() + 0.45
                            with self._lock:
                                self._state = "capturing"
                                snap_label = self._label
                                snap_mode  = self._mode
                                snap_n     = self._samples_per_trigger
                            flat_copy = flat.copy()
                            threading.Thread(
                                target=self._save_snapshot,
                                args=(flat_copy, snap_label, snap_mode, snap_n),
                                daemon=True,
                            ).start()

                    # Build status line for overlay
                    with self._lock:
                        cur_state  = self._state
                        cur_result = self._result
                        seq_count  = len(self._seq_rows) if self._recording else 0

                    if recording:
                        status_line = f"Recording: {seq_count} frames — trigger to stop"
                    elif cur_state == "done" and cur_result:
                        if "error" in cur_result:
                            status_line = f"Error: {cur_result['error']}"
                        else:
                            status_line = f"Saved {cur_result.get('generated', 0)} samples (total: {cur_result.get('total_for_label', 0)})"
                    elif cur_state == "capturing":
                        status_line = "Saving…"
                    else:
                        ct_label = "snapshot" if self._capture_type == "snapshot" else "sequence"
                        status_line = f"Press trigger for {ct_label}  ({spt} samples)" if self._capture_type == "snapshot" else "Press trigger to start recording"

                    self._live_jpeg = _encode_annotated(
                        frame, label, prediction, confidence, status_line
                    )
        finally:
            cap.release()
            self._live_jpeg = None
            with self._lock:
                if self._state not in ("done",):
                    self._state = "idle"

    # ── Save helpers (run in background threads) ──────────────────────────────

    def _save_snapshot(self, flat: np.ndarray, label: str, mode: str, n: int) -> None:
        try:
            if mode == "two_hand":
                from src.lab.augment import TwoHandAugmentationEngine
                engine    = TwoHandAugmentationEngine()
                synthetic = engine.generate(flat, n)

                npy_dir = _NPY_TWO_HAND_DIR / label
                npy_dir.mkdir(parents=True, exist_ok=True)

                needs_hdr = not _CSV_TWO_HAND_PATH.exists() or _CSV_TWO_HAND_PATH.stat().st_size == 0
                with open(_CSV_TWO_HAND_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if needs_hdr:
                        writer.writerow(TWO_HAND_HEADER)
                    for row in synthetic:
                        writer.writerow([label] + row.tolist())

                base_ts = int(time.time() * 1000)
                for i, row in enumerate(synthetic):
                    np.save(str(npy_dir / f"{base_ts + i}.npy"), row.astype("float32"))

                total = _csv_two_hand_label_counts().get(label, 0)
                with self._lock:
                    self._result = {"generated": n, "total_for_label": total, "mode": "snapshot"}
                    self._state  = "done"
            else:
                from src.lab.augment import AugmentationEngine
                engine    = AugmentationEngine()
                synthetic = engine.generate(flat, n)

                npy_dir = _NPY_DIR / label
                npy_dir.mkdir(parents=True, exist_ok=True)

                needs_hdr = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0
                with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if needs_hdr:
                        writer.writerow(HEADER)
                    for row in synthetic:
                        writer.writerow([label] + row.tolist())

                base_ts = int(time.time() * 1000)
                for i, row in enumerate(synthetic):
                    np.save(str(npy_dir / f"{base_ts + i}.npy"), row.astype("float32"))

                total = _csv_label_counts().get(label, 0)
                with self._lock:
                    self._result = {"generated": n, "total_for_label": total, "mode": "snapshot"}
                    self._state  = "done"
        except Exception as exc:
            with self._lock:
                self._result = {"error": str(exc)}
                self._state  = "done"

    def _save_sequence(self, rows: list[np.ndarray]) -> None:
        with self._lock:
            label = self._label
            mode  = self._mode

        try:
            csv_path     = _CSV_TWO_HAND_PATH if mode == "two_hand" else _CSV_PATH
            npy_dir_base = _NPY_TWO_HAND_DIR  if mode == "two_hand" else _NPY_DIR
            header       = TWO_HAND_HEADER    if mode == "two_hand" else HEADER

            npy_dir = npy_dir_base / label
            npy_dir.mkdir(parents=True, exist_ok=True)

            needs_hdr = not csv_path.exists() or csv_path.stat().st_size == 0
            saved = 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if needs_hdr:
                    writer.writerow(header)
                for flat in rows:
                    ts = int(time.time() * 1000) + saved
                    np.save(str(npy_dir / f"{ts}.npy"), flat.astype("float32"))
                    writer.writerow([label] + flat.tolist())
                    saved += 1

            count_fn = _csv_two_hand_label_counts if mode == "two_hand" else _csv_label_counts
            total = count_fn().get(label, 0)
            with self._lock:
                self._result = {"generated": saved, "total_for_label": total, "mode": "sequence"}
                self._state  = "done"
        except Exception as exc:
            with self._lock:
                self._result = {"error": str(exc)}
                self._state  = "done"


# Module-level singleton — one session per server process
_session = CameraSession()


# ── Stats & cameras ──────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    counts           = _csv_label_counts()
    two_hand_counts  = _csv_two_hand_label_counts()
    return {
        "labels":          counts,
        "total":           sum(counts.values()),
        "two_hand_labels": two_hand_counts,
        "two_hand_total":  sum(two_hand_counts.values()),
    }


@router.get("/cameras")
async def get_cameras():
    loop = asyncio.get_running_loop()
    cameras = await loop.run_in_executor(None, _probe_cameras)
    return {"cameras": cameras}


# ── Always-on annotated MJPEG stream ─────────────────────────────────────────

@router.get("/capture/stream")
async def stream():
    """Always-on MJPEG stream served from the CameraSession live buffer.
    Shows a placeholder when no session is running."""

    _ph = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(_ph, "Select a label to activate preview", (18, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 100), 1, cv2.LINE_AA)
    _, _ph_enc = cv2.imencode(".jpg", _ph)
    placeholder = _ph_enc.tobytes()

    async def _generate():
        while True:
            jpeg = _session.get_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + (jpeg if jpeg is not None else placeholder)
                + b"\r\n"
            )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Session management endpoints ──────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    camera:              int                          = 0
    label:               str
    capture_type:        Literal["snapshot", "sequence"] = "snapshot"
    mode:                Literal["single", "two_hand"]   = "single"
    samples_per_trigger: int = Field(default=50, ge=1, le=500)


@router.post("/session/start")
async def session_start(body: SessionStartRequest):
    if body.label not in _VALID_LABELS:
        raise HTTPException(422, f"Unknown label '{body.label}'.")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, _session.start,
        body.camera, body.label, body.capture_type, body.mode, body.samples_per_trigger,
    )
    return {"ok": True}


@router.post("/session/stop")
async def session_stop():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _session.stop)
    return {"ok": True}


@router.post("/session/trigger")
async def session_trigger():
    action = _session.trigger()
    return {"ok": True, "action": action}


@router.get("/session/status")
async def session_status():
    return _session.get_status()


class SessionLabelRequest(BaseModel):
    label: str


@router.post("/session/label")
async def session_set_label(body: SessionLabelRequest):
    if body.label not in _VALID_LABELS:
        raise HTTPException(422, f"Unknown label '{body.label}'.")
    _session.set_label(body.label)
    return {"ok": True}


# ── Zip upload ───────────────────────────────────────────────────────────────

def _do_upload(content: bytes) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    try:
        model_path = _ensure_model()
    except Exception as exc:
        return {"error": f"HandLandmarker model unavailable: {exc}"}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "upload.zip"
        zip_path.write_bytes(content)

        extract_dir = tmp_path / "dataset"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            return {"error": "Invalid zip file."}

        dataset_root = _find_dataset_root(extract_dir)
        if dataset_root is None:
            return {
                "error": "No recognized label folders found in zip. "
                         "Expected folders named after gesture tokens (STOP, PLAY, etc.)."
            }

        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            running_mode=mp_vision.RunningMode.IMAGE,
        )

        rows_written = 0
        rows_skipped = 0
        label_counts: dict[str, int] = {}

        _NPY_DIR.mkdir(parents=True, exist_ok=True)
        csv_needs_header = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0

        with mp_vision.HandLandmarker.create_from_options(options) as detector:
            with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if csv_needs_header:
                    writer.writerow(HEADER)

                for label_dir in sorted(dataset_root.iterdir()):
                    if not label_dir.is_dir():
                        continue
                    label = label_dir.name.upper()
                    if label not in _VALID_LABELS:
                        continue

                    out_dir = _NPY_DIR / label
                    out_dir.mkdir(exist_ok=True)

                    images = sorted(
                        list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))
                    )
                    for img_path in images:
                        bgr = cv2.imread(str(img_path))
                        if bgr is None:
                            rows_skipped += 1
                            continue
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = detector.detect(mp_img)
                        if not result.hand_landmarks:
                            rows_skipped += 1
                            continue
                        flat = normalize_landmarks(result.hand_landmarks[0])
                        ts = int(time.monotonic() * 1000)
                        np.save(
                            str(out_dir / f"upload_{ts}_{rows_written}.npy"),
                            flat.astype("float32"),
                        )
                        writer.writerow([label] + flat.tolist())
                        rows_written += 1
                        label_counts[label] = label_counts.get(label, 0) + 1

    return {"imported": rows_written, "skipped": rows_skipped, "by_label": label_counts}


@router.post("/upload")
async def upload_dataset(file: UploadFile):
    if not file.filename.endswith(".zip"):
        raise HTTPException(422, "File must be a .zip archive.")
    if file.size and file.size > 200 * 1024 * 1024:
        raise HTTPException(413, "Zip too large. Maximum 200 MB.")

    content = await file.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_upload, content)

    if "error" in result:
        code = 503 if "model" in result["error"].lower() else 422
        raise HTTPException(code, result["error"])
    return result


# ── Delete label ─────────────────────────────────────────────────────────────

@router.delete("/{label}")
async def delete_label_data(
    label: str,
    mode: Literal["single", "two_hand"] = Query(default="single"),
):
    if label not in _VALID_LABELS:
        raise HTTPException(422, f"Unknown label '{label}'.")

    csv_path     = _CSV_TWO_HAND_PATH if mode == "two_hand" else _CSV_PATH
    npy_dir_base = _NPY_TWO_HAND_DIR  if mode == "two_hand" else _NPY_DIR

    removed = 0
    if csv_path.exists():
        kept: list[list] = []
        header = None
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0] == label:
                    removed += 1
                else:
                    kept.append(row)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(kept)

    npy_dir = npy_dir_base / label
    if npy_dir.exists():
        shutil.rmtree(npy_dir)

    return {"deleted": removed, "label": label}
