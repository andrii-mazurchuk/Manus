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
from src.core.event_bus import EventBus
from src.core.gesture_event import GestureEvent, GestureToken
from src.adapters.websocket_adapter import WebSocketAdapter
from src.adapters.pc_adapter import PCAdapter
from src.adapters.mqtt_adapter import MQTTAdapter

MODEL_PATH = Path(__file__).parent / "src" / "models" / "hand_landmarker.task"


class TerminalAdapter(BaseAdapter):
    def on_gesture(self, event: GestureEvent) -> None:
        print(f"  {event.gesture.value:<10}  {event.confidence:.2%}", flush=True)


def run(camera_index: int, threshold: float) -> None:
    clf = GestureClassifier()
    bus = EventBus.get()
    bus.register(TerminalAdapter())
    bus.register(WebSocketAdapter())
    bus.register(PCAdapter())
    bus.register(MQTTAdapter())

    if not MODEL_PATH.exists():
        sys.exit(
            f"ERROR: hand_landmarker.task not found at {MODEL_PATH}.\n"
            "Run 'uv run src/data/extract_landmarks.py' once to download it."
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                label_str, conf = clf.predict(landmarks)

                if conf >= threshold:
                    try:
                        token = GestureToken(label_str)
                    except ValueError:
                        token = None

                    if token is not None:
                        label_index = int(clf._le.transform([label_str])[0])
                        bus.emit(GestureEvent(token, float(conf), label_index, time.time()))

            cv2.imshow("Manus -- Main Pipeline", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
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
