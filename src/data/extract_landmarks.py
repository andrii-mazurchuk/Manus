# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python",
#   "mediapipe",
#   "numpy",
# ]
# ///
"""
Extract MediaPipe hand landmarks from the LeapGestRecog dataset.

Reads from a local dataset folder, runs MediaPipe Hands on every image,
normalizes landmarks relative to the wrist, and writes rows to
data/gestures.csv matching the contract schema:

    label, x0, y0, x1, y1, ..., x20, y20

Expected dataset layout:
    data/leapGestRecog/
        00/
            01_palm/   frame_00_01_0001.png ...
            02_l/
            ...
        01/
        ...

Note: images are infrared (Leap Motion), converted to RGB for MediaPipe.
MediaPipe detection confidence is lowered to 0.3 to compensate.

Uses mediapipe 0.10+ Tasks API. The HandLandmarker model (~1 MB) is
downloaded automatically to models/hand_landmarker.task on first run.

Usage:
    uv run data/extract_landmarks.py
    uv run data/extract_landmarks.py --dataset data/leapGestRecog
"""

import argparse
import csv
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── HandLandmarker model (mediapipe 0.10+ Tasks API) ────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

# ── Label mapping: dataset folder name → project token ──────────────────────
GESTURE_MAP = {
    "01_palm":       "PLAY",
    "02_l":          "MODE",
    "03_fist":       "STOP",
    "04_fist_moved": "STOP",
    "05_thumb":      "CONFIRM",
    "06_index":      "UP",
    "07_ok":         "CUSTOM",
    "08_palm_moved": "PLAY",
    "09_c":          "CANCEL",
    "10_down":       "DOWN",
}

OUTPUT_CSV = Path(__file__).parent / "gestures.csv"

# Contract header: label + interleaved x/y per landmark (43 columns total)
HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]


def ensure_model() -> Path:
    """Download the HandLandmarker .task file if not already present."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HandLandmarker model (~1 MB) -> {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.\n")
    return MODEL_PATH


def normalize(landmarks) -> np.ndarray:
    """
    Translate all landmarks to wrist origin (landmark 0), then scale so the
    max absolute coordinate is 1.0 — maps everything into [-1, 1].

    landmarks: list of NormalizedLandmark (each has .x, .y attributes).
    Returns a flat array of shape (42,): [x0, y0, x1, y1, ..., x20, y20].
    """
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)  # (21, 2)
    coords -= coords[0]          # translate: wrist → origin
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale          # scale to [-1, 1]
    return coords.flatten()      # (42,)


def process_dataset(dataset_path: str) -> None:
    model_path = ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # lower threshold for infrared images
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    root = Path(dataset_path)
    rows_written = 0
    rows_skipped = 0
    skipped_by_gesture: dict[str, int] = {}
    label_counts: dict[str, int] = {}

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

            subject_dirs = sorted(d for d in root.iterdir() if d.is_dir())
            total_subjects = len(subject_dirs)

            for s_idx, subject_dir in enumerate(subject_dirs, 1):
                print(f"[{s_idx}/{total_subjects}] Subject: {subject_dir.name}", flush=True)

                gesture_dirs = sorted(d for d in subject_dir.iterdir() if d.is_dir())
                for gesture_dir in gesture_dirs:
                    folder_name = gesture_dir.name.lower()
                    label = GESTURE_MAP.get(folder_name)
                    if label is None:
                        print(f"  [skip] unmapped folder: {gesture_dir.name}")
                        continue

                    image_files = sorted(
                        list(gesture_dir.glob("*.png")) + list(gesture_dir.glob("*.jpg"))
                    )

                    for img_path in image_files:
                        bgr = cv2.imread(str(img_path))
                        if bgr is None:
                            rows_skipped += 1
                            continue

                        # MediaPipe Tasks API expects an mp.Image in SRGB format
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = detector.detect(mp_image)

                        if not result.hand_landmarks:
                            rows_skipped += 1
                            skipped_by_gesture[folder_name] = (
                                skipped_by_gesture.get(folder_name, 0) + 1
                            )
                            continue

                        flat = normalize(result.hand_landmarks[0])
                        writer.writerow([label] + flat.tolist())
                        rows_written += 1
                        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n-- Extraction complete ----------------------------------------------")
    print(f"  Rows written : {rows_written}")
    print(f"  Rows skipped (no hand detected): {rows_skipped}")
    if rows_written > 0:
        skip_pct = 100.0 * rows_skipped / (rows_written + rows_skipped)
        print(f"  Skip rate    : {skip_pct:.1f}%")
    print("\n  Samples per label:")
    for lbl, count in sorted(label_counts.items()):
        print(f"    {lbl:10s}: {count:5d}")
    if skipped_by_gesture:
        print("\n  Skipped breakdown (by gesture folder):")
        for name, count in sorted(skipped_by_gesture.items()):
            print(f"    {name}: {count}")
    print(f"\n  Output: {OUTPUT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks from LeapGestRecog dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "leapGestRecog",
        help="Path to the leapGestRecog root folder (default: data/leapGestRecog)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"ERROR: dataset folder not found: {args.dataset}\n"
            "Place the leapGestRecog folder at data/leapGestRecog or pass --dataset <path>"
        )

    print(f"Using dataset at: {args.dataset}\n")
    process_dataset(str(args.dataset))


if __name__ == "__main__":
    main()
