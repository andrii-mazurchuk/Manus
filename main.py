import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.core.base_adapter import BaseAdapter
from src.core.classifier import GestureClassifier
from src.core.two_hand_classifier import TwoHandGestureClassifier
from src.core.event_bus import EventBus
from src.core.gesture_event import GestureEvent, GestureToken
from src.core.sequence_recogniser import SequenceRecogniser
from src.adapters.websocket_adapter import WebSocketAdapter
from src.adapters.pc_adapter import PCAdapter
from src.adapters.mqtt_adapter import MQTTAdapter

MODEL_PATH = Path(__file__).parent / "src" / "models" / "hand_landmarker.task"


class TerminalAdapter(BaseAdapter):
    def on_gesture(self, event: GestureEvent) -> None:
        print(f"  {event.gesture.value:<10}  {event.confidence:.2%}", flush=True)


def _load_two_hand_classifier() -> TwoHandGestureClassifier | None:
    """Load two-hand classifier if trained; return None if model is absent."""
    try:
        return TwoHandGestureClassifier()
    except FileNotFoundError:
        print(
            "[main] Two-hand model not found — two-hand gestures disabled. "
            "Capture data and train via Studio to enable.",
            file=sys.stderr,
        )
        return None


def run(camera_index: int, threshold: float) -> None:
    clf = GestureClassifier()
    two_hand_clf = _load_two_hand_classifier()
    bus = EventBus.get()
    bus.register(TerminalAdapter())
    bus.register(WebSocketAdapter())
    bus.register(PCAdapter())
    bus.register(MQTTAdapter())
    bus.register(SequenceRecogniser())

    if not MODEL_PATH.exists():
        sys.exit(
            f"ERROR: hand_landmarker.task not found at {MODEL_PATH}.\n"
            "Run 'uv run src/data/extract_landmarks.py' once to download it."
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                cv2.imshow("Manus -- Main Pipeline", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue

            # Sort detected hands by handedness: right = primary, left = secondary.
            right_hand = None
            left_hand = None
            for i, hand_lms in enumerate(result.hand_landmarks):
                side = result.handedness[i][0].category_name  # "Right" or "Left"
                if side == "Right":
                    right_hand = hand_lms
                else:
                    left_hand = hand_lms

            primary = right_hand or left_hand
            secondary = left_hand if (right_hand is not None) else None

            emitted = False

            # Two-hand classifier: runs when model is loaded (handles both
            # one-hand and two-hand gestures via zero-masking).
            if two_hand_clf is not None:
                label_str, conf = two_hand_clf.predict(primary, secondary)
                if conf >= threshold:
                    try:
                        token = GestureToken(label_str)
                        label_index = int(two_hand_clf._le.transform([label_str])[0])
                        bus.emit(GestureEvent(token, float(conf), label_index, time.time()))
                        emitted = True
                    except ValueError:
                        pass

            # Single-hand classifier: fallback when two-hand model absent or
            # when two-hand model produced low confidence.
            if not emitted:
                label_str, conf = clf.predict(primary)
                if conf >= threshold:
                    try:
                        token = GestureToken(label_str)
                        label_index = int(clf._le.transform([label_str])[0])
                        bus.emit(GestureEvent(token, float(conf), label_index, time.time()))
                    except ValueError:
                        pass

            cv2.imshow("Manus -- Main Pipeline", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Manus production pipeline")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold")
    args = parser.parse_args()
    run(args.camera, args.threshold)


if __name__ == "__main__":
    main()
