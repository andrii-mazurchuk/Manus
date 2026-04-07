"""
Live webcam gesture recognition pipeline.

Webcam -> MediaPipe HandLandmarker -> GestureClassifier -> terminal + overlay.

Controls:
    Q / ESC  quit

Usage:
    python capture.py
    python capture.py --camera 1   # if default camera index 0 doesn't work
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Make src/core importable when running as a top-level script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.classifier import GestureClassifier

MODEL_PATH = Path(__file__).parent.parent / "src" / "models" / "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.70

# Hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                   # palm base
]

# BGR colours
COLOR_LANDMARK   = (0, 220, 255)   # yellow
COLOR_CONNECTION = (200, 200, 200) # light grey
COLOR_ABOVE      = (0, 220, 0)     # green  — confidence >= threshold
COLOR_BELOW      = (140, 140, 140) # grey   — confidence < threshold


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    """Draw hand skeleton onto frame in-place. landmarks: list of NormalizedLandmark."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_CONNECTION, 1, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, label: str, confidence: float) -> None:
    """Overlay gesture label and confidence bar at the top of the frame."""
    h, w = frame.shape[:2]
    color = COLOR_ABOVE if confidence >= CONFIDENCE_THRESHOLD else COLOR_BELOW

    # Label text
    text = f"{label}  {confidence:.0%}"
    cv2.putText(frame, text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color,    2, cv2.LINE_AA)

    # Confidence bar
    bar_w = int((w - 32) * confidence)
    cv2.rectangle(frame, (16, 60), (16 + bar_w, 72), color, -1)
    cv2.rectangle(frame, (16, 60), (w - 16, 72),     (80, 80, 80), 1)

    if confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(
            frame, "below threshold", (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BELOW, 1, cv2.LINE_AA,
        )


def run(camera_index: int) -> None:
    print("Loading classifier...")
    clf = GestureClassifier()
    print(f"  Classes: {clf.classes}")

    if not MODEL_PATH.exists():
        sys.exit(
            f"ERROR: hand_landmarker.task not found at {MODEL_PATH}.\n"
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

    print(f"\nWebcam opened (index {camera_index}). Press Q or ESC to quit.\n")

    last_label = ""
    last_conf  = 0.0
    fps_t      = time.perf_counter()
    fps        = 0.0
    frame_n    = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: dropped frame, retrying...")
                continue

            frame_n += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)

                label, conf = clf.predict(landmarks)

                if label != last_label or frame_n % 30 == 0:
                    marker = "" if conf >= CONFIDENCE_THRESHOLD else " (low conf)"
                    print(f"  {label:<10}  {conf:.2%}{marker}", flush=True)

                last_label = label
                last_conf  = conf

            if last_label:
                draw_label(frame, last_label, last_conf)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
            fps_t = now
            cv2.putText(
                frame, f"FPS {fps:.0f}", (frame.shape[1] - 80, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
            )

            cv2.imshow("Manus -- Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("\nCamera released. Bye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live gesture recognition")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0)",
    )
    args = parser.parse_args()
    run(args.camera)


if __name__ == "__main__":
    main()
