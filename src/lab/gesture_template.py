"""
Webcam template capture for the synthetic augmentation pipeline.

Captures a burst of hand-landmark frames from the webcam, averages them into
a single canonical "template" array, and saves it to src/lab/templates/.

The template is the seed that AugmentationEngine.generate() expands into
hundreds of synthetic training rows.

Usage:
    # As context manager (recommended):
    with GestureTemplateCapture() as capturer:
        template = capturer.capture("WAVE")   # opens webcam UI

    # Load an existing template (no webcam needed):
    capturer = GestureTemplateCapture()
    template = capturer.load("WAVE")           # (42,) float32
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_ROOT      = Path(__file__).parent.parent               # src/
SRC_MODELS    = SRC_ROOT / "models"
TEMPLATES_DIR = Path(__file__).parent / "templates"        # src/lab/templates/
MODEL_PATH    = SRC_MODELS / "hand_landmarker.task"

sys.path.insert(0, str(SRC_ROOT))
from core.normalizer import normalize_landmarks, normalize_coords, HAND_CONNECTIONS  # noqa: E402

COLOR_LANDMARK   = (0, 220, 255)    # yellow
COLOR_CONNECTION = (200, 200, 200)  # light grey


def _draw_landmarks(frame: np.ndarray, landmarks) -> None:
    """Draw hand skeleton on frame in-place."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_CONNECTION, 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GestureTemplateCapture:
    """
    Captures a short burst of webcam frames and averages them into a landmark
    template for the AugmentationEngine.

    The webcam window shows live landmark overlay and guides the user through
    the capture with on-screen instructions.

    Args:
        camera_index: OpenCV camera index (default 0).
        n_frames:     Number of consecutive frames to average (default 5).
        model_path:   Path to the MediaPipe HandLandmarker .task file.
    """

    def __init__(
        self,
        camera_index: int = 0,
        n_frames: int = 5,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.camera_index = camera_index
        self.n_frames     = n_frames
        self.model_path   = model_path
        self._cap: cv2.VideoCapture | None       = None
        self._detector: mp_vision.HandLandmarker | None = None

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Initialise webcam and MediaPipe detector (idempotent)."""
        if self._cap is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"HandLandmarker model not found at {self.model_path}.\n"
                "Run: uv run src/data/extract_landmarks.py  (downloads it automatically)"
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.model_path)
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._detector.close()
            self._detector = None
            self._cap = None
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                f"Try --camera 1, or close any other app using the webcam."
            )

    def close(self) -> None:
        """Release webcam and detector."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        cv2.destroyAllWindows()

    def __enter__(self) -> "GestureTemplateCapture":
        self._open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, label: str) -> np.ndarray:
        """
        Open a webcam window and guide the user through capturing a template.

        The user holds the gesture and presses SPACE.  The tool then collects
        n_frames consecutive frames with a detected hand, averages their
        normalized landmark arrays, renormalizes the result, saves it to
        src/lab/templates/<label>.npy, and returns it.

        Args:
            label: Gesture token name (e.g. "WAVE", "PINCH", "STOP").

        Returns:
            (42,) float32 normalized template array.

        Raises:
            FileNotFoundError: If the HandLandmarker model is missing.
            RuntimeError:      If the webcam cannot be opened.
        """
        self._open()

        WINDOW = f"Manus — Template Capture [{label}]"
        BURST_TIMEOUT = 3.0   # seconds to wait for n_frames during burst

        # State machine states
        WAITING  = "waiting"
        BURSTING = "bursting"
        DONE     = "done"

        state               = WAITING
        burst_frames: list[np.ndarray] = []
        burst_start         = 0.0
        current_landmarks   = None
        flash_counter       = 0     # frames to show "Captured!" overlay
        warning_counter     = 0     # frames to show warning text
        warning_text        = ""
        template: np.ndarray | None = None

        while state != DONE:
            ok, frame = self._cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self._detector.detect(mp_image)

            if result.hand_landmarks:
                current_landmarks = result.hand_landmarks[0]
                _draw_landmarks(frame, current_landmarks)
            else:
                current_landmarks = None

            # ── state logic ─────────────────────────────────────────────
            if state == WAITING:
                self._draw_waiting_ui(frame, label, warning_counter, warning_text)
                if warning_counter > 0:
                    warning_counter -= 1

            elif state == BURSTING:
                elapsed = time.time() - burst_start

                if current_landmarks is not None:
                    flat = normalize_landmarks(current_landmarks)
                    burst_frames.append(flat)

                if len(burst_frames) >= self.n_frames:
                    # Success — average and renormalize
                    stack    = np.stack(burst_frames, axis=0)   # (n, 42)
                    mean_pts = stack.mean(axis=0).reshape(21, 2).astype(np.float32)
                    template = normalize_coords(mean_pts)

                    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(TEMPLATES_DIR / f"{label}.npy", template)

                    state         = WAITING   # briefly show "Captured!" before DONE
                    flash_counter = 40
                    burst_frames  = []

                elif elapsed > BURST_TIMEOUT:
                    warning_text    = "Burst incomplete — hold still, press SPACE again"
                    warning_counter = 60
                    burst_frames    = []
                    state           = WAITING

                else:
                    self._draw_burst_ui(
                        frame, label,
                        len(burst_frames), self.n_frames, elapsed, BURST_TIMEOUT
                    )

            # ── flash overlay (post-capture) ─────────────────────────────
            if flash_counter > 0:
                self._draw_captured_overlay(frame)
                flash_counter -= 1
                if flash_counter == 0:
                    state = DONE

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):   # Q or ESC → abort
                self.close()
                raise KeyboardInterrupt("Template capture aborted by user.")

            if key == 32 and state == WAITING:     # SPACE
                if current_landmarks is None:
                    warning_text    = "No hand in frame — hold gesture and try again"
                    warning_counter = 45
                else:
                    state       = BURSTING
                    burst_start = time.time()
                    burst_frames = []

        cv2.destroyWindow(WINDOW)
        assert template is not None
        print(f"  Template saved → {TEMPLATES_DIR / label}.npy")
        return template

    def load(self, label: str) -> np.ndarray:
        """
        Load a previously captured template from disk.

        Args:
            label: Gesture token name.

        Returns:
            (42,) float32 normalized template array.

        Raises:
            FileNotFoundError: If no template exists for this label.
        """
        path = TEMPLATES_DIR / f"{label}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"No template found for '{label}' at {path}.\n"
                f"Run without --no-capture to record it first."
            )
        arr = np.load(path)
        if arr.shape != (42,):
            raise ValueError(
                f"Template at {path} has unexpected shape {arr.shape} (expected (42,))."
            )
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_waiting_ui(
        frame: np.ndarray,
        label: str,
        warning_counter: int,
        warning_text: str,
    ) -> None:
        h = frame.shape[0]

        # Instruction lines
        cv2.putText(
            frame,
            f"Hold [{label.upper()}] gesture",
            (16, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press SPACE to capture  |  Q to quit",
            (16, 76),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Warning text (fades by frame count)
        if warning_counter > 0:
            cv2.putText(
                frame,
                warning_text,
                (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 220), 1, cv2.LINE_AA,
            )

    @staticmethod
    def _draw_burst_ui(
        frame: np.ndarray,
        label: str,
        collected: int,
        total: int,
        elapsed: float,
        timeout: float,
    ) -> None:
        cv2.putText(
            frame,
            f"Capturing [{label}]... {collected}/{total}",
            (16, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 120), 2, cv2.LINE_AA,
        )
        # Timeout progress bar
        w = frame.shape[1]
        bar_w = int((w - 32) * min(elapsed / timeout, 1.0))
        cv2.rectangle(frame, (16, 58), (16 + bar_w, 68), (0, 180, 80), -1)
        cv2.rectangle(frame, (16, 58), (w - 16, 68),     (60, 60, 60), 1)

    @staticmethod
    def _draw_captured_overlay(frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 200, 80), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "Captured!",
            (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA,
        )
