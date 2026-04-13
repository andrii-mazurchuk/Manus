import asyncio
import csv
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.gesture_event import GestureToken
from src.core.normalizer import normalize_landmarks

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# ── Paths (relative to repo root — where uvicorn is launched) ────────────────
_CSV_PATH   = Path("src/data/gestures.csv")
_NPY_DIR    = Path("src/data/gestures")
_MODEL_PATH = Path("src/models/hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

_capture_lock = threading.Lock()

_VALID_LABELS = {t.value for t in GestureToken}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_model() -> Path:
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
        next(reader, None)  # skip header
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
    """Return the directory that directly contains label-named subdirs."""
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


# ── Stats & cameras ──────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    counts = _csv_label_counts()
    return {"labels": counts, "total": sum(counts.values())}


@router.get("/cameras")
async def get_cameras():
    loop = asyncio.get_running_loop()
    cameras = await loop.run_in_executor(None, _probe_cameras)
    return {"cameras": cameras}


# ── MJPEG preview stream ─────────────────────────────────────────────────────

@router.get("/capture/stream")
async def stream(camera: int = 0):
    """Live MJPEG stream from the selected camera for positioning preview."""

    async def _generate():
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            return
        try:
            while True:
                ok, frame = await loop.run_in_executor(None, cap.read)
                if not ok:
                    await asyncio.sleep(0.05)
                    continue
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                await asyncio.sleep(0.033)  # ~30 fps
        finally:
            cap.release()

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Live capture ─────────────────────────────────────────────────────────────

class CaptureRequest(BaseModel):
    camera: int = 0
    label: str
    count: int = Field(default=20, ge=1, le=500)


def _do_capture(camera_index: int, label: str, count: int) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    try:
        model_path = _ensure_model()
    except Exception as exc:
        return {"error": f"Could not load HandLandmarker model: {exc}"}

    # Retry camera open to handle timing if MJPEG stream just released it
    cap = None
    for _ in range(6):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
        cap.release()
        time.sleep(0.3)
    else:
        return {"error": f"Could not open camera {camera_index}. Is another process using it?"}

    captured = 0
    skipped = 0
    start = time.monotonic()
    timeout = count * 3 + 15  # generous: 3 s per sample + 15 s buffer

    _NPY_DIR.mkdir(parents=True, exist_ok=True)
    label_dir = _NPY_DIR / label
    label_dir.mkdir(exist_ok=True)

    csv_needs_header = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0

    base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as detector:
            with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if csv_needs_header:
                    writer.writerow(HEADER)

                while captured < count and (time.monotonic() - start) < timeout:
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_img)

                    if not result.hand_landmarks:
                        skipped += 1
                        continue

                    flat = normalize_landmarks(result.hand_landmarks[0])
                    ts = int(time.monotonic() * 1000)
                    np.save(str(label_dir / f"{ts}_{captured}.npy"), flat.astype("float32"))
                    writer.writerow([label] + flat.tolist())
                    captured += 1
    finally:
        cap.release()

    return {
        "captured": captured,
        "skipped": skipped,
        "total_for_label": _csv_label_counts().get(label, 0),
    }


@router.post("/capture")
async def capture(body: CaptureRequest):
    if body.label not in _VALID_LABELS:
        raise HTTPException(422, f"Unknown label '{body.label}'. Valid: {sorted(_VALID_LABELS)}")

    if not _capture_lock.acquire(blocking=False):
        raise HTTPException(409, "A capture session is already running. Wait for it to finish.")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_capture, body.camera, body.label, body.count)
    finally:
        _capture_lock.release()

    if "error" in result:
        raise HTTPException(503, result["error"])
    return result


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
            min_hand_detection_confidence=0.3,  # lower for varied image conditions
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
async def delete_label_data(label: str):
    if label not in _VALID_LABELS:
        raise HTTPException(422, f"Unknown label '{label}'.")

    removed = 0
    if _CSV_PATH.exists():
        kept: list[list] = []
        header = None
        with open(_CSV_PATH, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0] == label:
                    removed += 1
                else:
                    kept.append(row)
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(kept)

    npy_dir = _NPY_DIR / label
    if npy_dir.exists():
        shutil.rmtree(npy_dir)

    return {"deleted": removed, "label": label}
