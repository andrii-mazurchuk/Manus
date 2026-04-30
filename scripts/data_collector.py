"""
Interactive webcam data collector for gesture training.

Shows live landmark overlay. Press a key to snapshot the current hand pose
and save it as a training sample.

Key bindings:
    P  ->  PLAY
    S  ->  STOP
    U  ->  UP
    D  ->  DOWN
    C  ->  CONFIRM
    X  ->  CANCEL
    M  ->  MODE
    T  ->  CUSTOM
    Q / ESC  ->  quit

Each capture:
  - saves a .npy file to  src/data/gestures/<LABEL>/
  - appends a row to      src/data/gestures.csv

Sample counts in the UI are read live from the per-label directory file counts.

Usage:
    python data_collector.py
    python data_collector.py --camera 1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.normalizer import normalize_landmarks, HAND_CONNECTIONS  # noqa: E402

MODEL_PATH   = Path(__file__).parent.parent / "src" / "models" / "hand_landmarker.task"
GESTURES_DIR = Path(__file__).parent.parent / "src" / "data" / "gestures"
CSV_PATH     = Path(__file__).parent.parent / "src" / "data" / "gestures.csv"

CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

KEY_MAP: dict[int, str] = {
    ord("p"): "PLAY",    ord("P"): "PLAY",
    ord("s"): "STOP",    ord("S"): "STOP",
    ord("u"): "UP",      ord("U"): "UP",
    ord("d"): "DOWN",    ord("D"): "DOWN",
    ord("c"): "CONFIRM", ord("C"): "CONFIRM",
    ord("x"): "CANCEL",  ord("X"): "CANCEL",
    ord("m"): "MODE",    ord("M"): "MODE",
    ord("t"): "CUSTOM",  ord("T"): "CUSTOM",
}

KEY_HINT: dict[str, str] = {
    "PLAY": "P", "STOP": "S", "UP": "U", "DOWN": "D",
    "CONFIRM": "C", "CANCEL": "X", "MODE": "M", "CUSTOM": "T",
}

ALL_LABELS = ["PLAY", "STOP", "UP", "DOWN", "CONFIRM", "CANCEL", "MODE", "CUSTOM"]

TARGET_SAMPLES = 100   # per label — green when reached

# BGR colours
COLOR_LANDMARK   = (0, 220, 255)   # yellow
COLOR_CONNECTION = (200, 200, 200) # light grey



def count_samples() -> dict[str, int]:
    """Count .npy files in each per-label directory."""
    return {
        label: len(list((GESTURES_DIR / label).glob("*.npy")))
        if (GESTURES_DIR / label).exists() else 0
        for label in ALL_LABELS
    }


def save_sample(label: str, flat: np.ndarray) -> None:
    """Save .npy file and append a row to gestures.csv."""
    label_dir = GESTURES_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    np.save(label_dir / f"{ts}.npy", flat)

    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([label] + flat.tolist())


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_CONNECTION, 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)


def draw_ui(
    frame: np.ndarray,
    counts: dict[str, int],
    last_label: str,
    hand_present: bool,
    flash: float,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 230
    panel_x = w - panel_w

    # semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # --- header ---
    cv2.putText(frame, "GESTURE SAMPLES", (panel_x + 8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (panel_x + 8, 33), (w - 8, 33), (60, 60, 60), 1)

    # --- per-label rows ---
    row_h = 36
    for i, label in enumerate(ALL_LABELS):
        top = 42 + i * row_h
        count = counts[label]
        is_active = label == last_label

        if count >= TARGET_SAMPLES:
            bar_color = (0, 200, 80)     # green — target met
        elif count >= TARGET_SAMPLES // 2:
            bar_color = (0, 180, 220)    # cyan  — halfway
        else:
            bar_color = (100, 100, 100)  # grey  — early

        text_color = (0, 255, 255) if is_active else bar_color

        key = KEY_HINT.get(label, "?")
        cv2.putText(frame, f"[{key}] {label}", (panel_x + 8, top + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(count), (w - 38, top + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

        # progress bar
        bar_area_w = panel_w - 16
        filled = min(int(bar_area_w * count / TARGET_SAMPLES), bar_area_w)
        cv2.rectangle(frame, (panel_x + 8, top + 18), (panel_x + 8 + filled, top + 26),
                      bar_color, -1)
        cv2.rectangle(frame, (panel_x + 8, top + 18), (panel_x + 8 + bar_area_w, top + 26),
                      (50, 50, 50), 1)

    # --- footer ---
    total = sum(counts.values())
    complete = sum(1 for c in counts.values() if c >= TARGET_SAMPLES)
    cv2.line(frame, (panel_x + 8, h - 48), (w - 8, h - 48), (60, 60, 60), 1)
    cv2.putText(frame, f"Total: {total}  Done: {complete}/{len(ALL_LABELS)}",
                (panel_x + 8, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160),
                1, cv2.LINE_AA)

    # --- status bar at bottom of video area ---
    if hand_present:
        status_text = "Hand detected — press key to capture"
        status_color = (0, 220, 120)
    else:
        status_text = "No hand detected"
        status_color = (60, 60, 200)

    cv2.putText(frame, status_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)

    # --- flash overlay on capture ---
    if flash > 0.0:
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, 0), (panel_x, h), (0, 255, 120), -1)
        cv2.addWeighted(flash_overlay, flash * 0.35, frame, 1 - flash * 0.35, 0, frame)


def run(camera_index: int) -> None:
    if not MODEL_PATH.exists():
        sys.exit(
            f"ERROR: hand_landmarker.task not found at {MODEL_PATH}\n"
            "Run 'python src/data/extract_landmarks.py' once to download it."
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(
            f"ERROR: could not open camera index {camera_index}. "
            "Try --camera 1 or check that no other app is using the webcam."
        )

    print("Manus data collector ready.")
    print("Keys: P=PLAY  S=STOP  U=UP  D=DOWN  C=CONFIRM  X=CANCEL  M=MODE  T=CUSTOM  Q=quit")
    print(f"Target: {TARGET_SAMPLES} samples per gesture\n")

    current_landmarks = None
    last_label = ""
    flash = 0.0
    counts = count_samples()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                current_landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, current_landmarks)
            else:
                current_landmarks = None

            flash = max(0.0, flash - 0.07)
            draw_ui(frame, counts, last_label, current_landmarks is not None, flash)

            cv2.imshow("Manus -- Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in KEY_MAP:
                if current_landmarks is not None:
                    label = KEY_MAP[key]
                    flat = normalize_landmarks(current_landmarks)
                    save_sample(label, flat)
                    counts[label] += 1
                    last_label = label
                    flash = 1.0
                    print(f"  Captured {label:<8}  ({counts[label]} samples)", flush=True)
                else:
                    print("  No hand in frame — hold your hand steady first", flush=True)

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        total = sum(counts.values())
        complete = sum(1 for c in counts.values() if c >= TARGET_SAMPLES)
        print(f"\nSession done. {total} total samples, {complete}/{len(ALL_LABELS)} gestures at target.")
        print(f"Data directory : {GESTURES_DIR}")
        print(f"CSV            : {CSV_PATH}")
        print("Next step      : python train.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect gesture training data from webcam")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    args = parser.parse_args()
    run(args.camera)


if __name__ == "__main__":
    main()
