from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QFrame, QSizePolicy,
)

from app.config_manager import ConfigManager
from src.core.normalizer import HAND_CONNECTIONS

_PROJECT_ROOT      = Path(__file__).parent.parent.parent
_MODEL_PATH        = _PROJECT_ROOT / "src" / "models" / "hand_landmarker.task"
_CLASSIFIER_PATH   = _PROJECT_ROOT / "src" / "models" / "classifier.pkl"
_DEFAULT_THRESHOLD = 0.70

_COLOR_GREEN  = (0, 224, 118)    # BGR — above threshold
_COLOR_ORANGE = (32, 128, 224)   # BGR — below threshold (orange in BGR space)


# ── Widget helpers ────────────────────────────────────────────────────────────

def _primary_btn(text: str, h: int = 34, w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    if w:
        btn.setFixedWidth(w)
    btn.setStyleSheet("""
        QPushButton {
            background: #1a4a7a; color: #ffffff; border: none; border-radius: 5px;
            font-size: 13px; padding: 0 14px;
        }
        QPushButton:hover   { background: #1f5a90; }
        QPushButton:pressed { background: #163d64; }
        QPushButton:disabled { background: #2a2a2a; color: #555555; }
    """)
    return btn


def _ghost_btn(text: str, h: int = 34, w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    if w:
        btn.setFixedWidth(w)
    btn.setStyleSheet("""
        QPushButton {
            background: transparent; color: #aaaaaa;
            border: 1px solid #3a3a3a; border-radius: 5px;
            font-size: 13px; padding: 0 12px;
        }
        QPushButton:hover   { background: #2a2a2a; color: #dddddd; border-color: #555; }
        QPushButton:pressed { background: #1e1e1e; }
        QPushButton:disabled { color: #444; border-color: #2a2a2a; }
    """)
    return btn


def _section_lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size: 11px; font-weight: 600; color: #777777; letter-spacing: 1px; border: none;"
    )
    return lbl


# ── Drawing helper ────────────────────────────────────────────────────────────

def _draw_hand(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 160, 160), 1, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, (0, 220, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)


# ── Camera + inference worker ─────────────────────────────────────────────────

class _CameraWorker(QObject):
    # label="" means no confident prediction (no hand, or below threshold)
    frame_ready    = Signal(bytes, str, float)
    error_occurred = Signal(str)
    finished       = Signal()

    def __init__(
        self,
        camera_index: int,
        thresholds: dict[str, float],
        default_threshold: float,
    ) -> None:
        super().__init__()
        self._camera_index      = camera_index
        self._thresholds        = thresholds
        self._default_threshold = default_threshold
        self._running           = False

    @Slot()
    def run(self) -> None:
        self._running = True
        try:
            self._loop()
        finally:
            self._running = False
            self.finished.emit()

    def request_stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        if not _MODEL_PATH.exists():
            self.error_occurred.emit(
                "hand_landmarker.task not found.\n"
                "Run: uv run src/data/extract_landmarks.py"
            )
            return

        if not _CLASSIFIER_PATH.exists():
            self.error_occurred.emit(
                "classifier.pkl not found.\n"
                "Train a model first via the Train page."
            )
            return

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from src.core.classifier import GestureClassifier

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        detector   = mp_vision.HandLandmarker.create_from_options(options)
        classifier = GestureClassifier(_CLASSIFIER_PATH)
        cap        = cv2.VideoCapture(self._camera_index)

        if not cap.isOpened():
            detector.close()
            self.error_occurred.emit(
                f"Cannot open camera {self._camera_index}. "
                "Try a different index."
            )
            return

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    continue

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                out_label = ""
                out_conf  = 0.0

                if result.hand_landmarks:
                    lms   = result.hand_landmarks[0]
                    _draw_hand(frame, lms)

                    label, conf = classifier.predict(lms)
                    threshold   = self._thresholds.get(label, self._default_threshold)
                    out_conf    = conf

                    if conf >= threshold:
                        out_label = label
                        color     = _COLOR_GREEN
                    else:
                        color = _COLOR_ORANGE

                    cv2.putText(
                        frame,
                        f"{label}  {conf:.0%}",
                        (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.frame_ready.emit(jpeg.tobytes(), out_label, out_conf)
        finally:
            cap.release()
            detector.close()


# ── Page ──────────────────────────────────────────────────────────────────────

class TestPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg                          = ConfigManager()
        self._thresholds: dict[str, float] = {}
        self._thread: QThread | None       = None
        self._worker: _CameraWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Test")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        hdr.addWidget(title)
        hdr.addStretch()
        root.addLayout(hdr)
        root.addSpacing(4)

        subtitle = QLabel(
            "Live camera feed with real-time single-gesture classification. "
            "Train a model first, then start the camera."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: #666666;")
        root.addWidget(subtitle)
        root.addSpacing(14)

        # ── Body ──────────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)

        # Left: camera feed
        left = QVBoxLayout()
        left.setSpacing(8)

        self._cam_label = QLabel("Camera not started")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._cam_label.setMinimumSize(480, 340)
        self._cam_label.setStyleSheet(
            "background: #111111; border: 1px solid #2d2d2d; "
            "border-radius: 6px; color: #444444; font-size: 14px;"
        )
        left.addWidget(self._cam_label, stretch=1)

        self._cam_status_lbl = QLabel("Camera not started")
        self._cam_status_lbl.setStyleSheet("font-size: 12px; color: #555555;")
        left.addWidget(self._cam_status_lbl)

        body.addLayout(left, stretch=1)

        # Right: controls + prediction panel
        body.addWidget(self._build_right_panel())

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 6px;"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Model status ──────────────────────────────────────────────────────
        lay.addWidget(_section_lbl("MODEL"))
        self._model_status_lbl = QLabel("Checking…")
        self._model_status_lbl.setStyleSheet("font-size: 12px; color: #888888; border: none;")
        self._model_status_lbl.setWordWrap(True)
        lay.addWidget(self._model_status_lbl)

        lay.addWidget(_hline())

        # ── Camera controls ───────────────────────────────────────────────────
        lay.addWidget(_section_lbl("CAMERA"))

        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)
        idx_lbl = QLabel("Index:")
        idx_lbl.setStyleSheet("font-size: 12px; color: #aaaaaa; border: none;")
        cam_row.addWidget(idx_lbl)

        self._camera_spin = QSpinBox()
        self._camera_spin.setRange(0, 9)
        self._camera_spin.setValue(0)
        self._camera_spin.setFixedSize(56, 30)
        self._camera_spin.setStyleSheet("""
            QSpinBox {
                background: #252525; border: 1px solid #383838; border-radius: 4px;
                color: #dddddd; font-size: 12px; padding: 0 4px;
            }
            QSpinBox:focus { border-color: #4a9eff; }
        """)
        cam_row.addWidget(self._camera_spin)
        cam_row.addStretch()
        lay.addLayout(cam_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._start_btn = _primary_btn("Start", h=32)
        self._start_btn.clicked.connect(self._start_camera)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = _ghost_btn("Stop", h=32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_camera)
        btn_row.addWidget(self._stop_btn)
        lay.addLayout(btn_row)

        lay.addWidget(_hline())

        # ── Prediction display ────────────────────────────────────────────────
        lay.addWidget(_section_lbl("PREDICTION"))

        self._pred_label_lbl = QLabel("—")
        self._pred_label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pred_label_lbl.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #444444; border: none;"
        )
        lay.addWidget(self._pred_label_lbl)

        self._pred_conf_lbl = QLabel("")
        self._pred_conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pred_conf_lbl.setStyleSheet("font-size: 14px; color: #666666; border: none;")
        lay.addWidget(self._pred_conf_lbl)

        self._pred_threshold_lbl = QLabel("")
        self._pred_threshold_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pred_threshold_lbl.setStyleSheet("font-size: 11px; color: #444444; border: none;")
        lay.addWidget(self._pred_threshold_lbl)

        lay.addStretch()
        return panel

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_model_status()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._stop_camera()

    # ── Model status ──────────────────────────────────────────────────────────

    def _refresh_model_status(self) -> None:
        gestures = self._cfg.load_gesture_labels()
        self._thresholds = {
            g["name"]: float(g.get("threshold", _DEFAULT_THRESHOLD))
            for g in gestures
        }

        if not _CLASSIFIER_PATH.exists():
            self._model_status_lbl.setText("● No model — train one first")
            self._model_status_lbl.setStyleSheet(
                "font-size: 12px; color: #e05050; border: none;"
            )
            if self._worker is None:
                self._start_btn.setEnabled(False)
        else:
            n = len(gestures)
            self._model_status_lbl.setText(
                f"● Ready  ({n} gesture{'s' if n != 1 else ''})"
            )
            self._model_status_lbl.setStyleSheet(
                "font-size: 12px; color: #50d090; border: none;"
            )
            if self._worker is None:
                self._start_btn.setEnabled(True)

    # ── Camera management ─────────────────────────────────────────────────────

    def _start_camera(self) -> None:
        self._stop_camera()
        self._refresh_model_status()

        self._thread = QThread()
        self._worker = _CameraWorker(
            self._camera_spin.value(),
            dict(self._thresholds),
            _DEFAULT_THRESHOLD,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error_occurred.connect(self._on_camera_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._set_cam_status("Starting…", level=0)

    def _stop_camera(self) -> None:
        if self._worker:
            self._worker.request_stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None

        self._start_btn.setEnabled(_CLASSIFIER_PATH.exists())
        self._stop_btn.setEnabled(False)
        self._cam_label.clear()
        self._cam_label.setText("Camera not started")
        self._set_cam_status("Camera stopped.", level=0)
        self._clear_prediction()

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(bytes, str, float)
    def _on_frame_ready(self, jpeg: bytes, label: str, confidence: float) -> None:
        img = QImage.fromData(jpeg)
        if not img.isNull():
            pix = QPixmap.fromImage(img).scaled(
                self._cam_label.width(),
                self._cam_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cam_label.setPixmap(pix)

        if label:
            threshold = self._thresholds.get(label, _DEFAULT_THRESHOLD)
            self._pred_label_lbl.setText(label)
            self._pred_label_lbl.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #00e076; border: none;"
            )
            self._pred_conf_lbl.setText(f"{confidence:.1%}")
            self._pred_conf_lbl.setStyleSheet(
                "font-size: 14px; color: #aaaaaa; border: none;"
            )
            self._pred_threshold_lbl.setText(f"threshold {threshold:.0%}")
            self._set_cam_status("Hand detected  ✓", level=1)
        elif confidence > 0:
            # hand present but below threshold
            self._pred_label_lbl.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #e08020; border: none;"
            )
            self._pred_conf_lbl.setText(f"{confidence:.1%}  (below threshold)")
            self._pred_conf_lbl.setStyleSheet(
                "font-size: 14px; color: #888888; border: none;"
            )
            self._set_cam_status("Hand detected — below threshold", level=0)
        else:
            self._clear_prediction()
            self._set_cam_status("No hand in frame", level=-1)

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._stop_camera()
        self._set_cam_status(f"Error: {msg}", level=-1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear_prediction(self) -> None:
        self._pred_label_lbl.setText("—")
        self._pred_label_lbl.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #444444; border: none;"
        )
        self._pred_conf_lbl.setText("")
        self._pred_threshold_lbl.setText("")

    def _set_cam_status(self, msg: str, *, level: int) -> None:
        color = {1: "#50d090", 0: "#555555", -1: "#e05050"}.get(level, "#555555")
        self._cam_status_lbl.setText(msg)
        self._cam_status_lbl.setStyleSheet(f"font-size: 12px; color: {color};")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _hline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet(
        "border: none; max-height: 1px; background: #2d2d2d;"
    )
    return sep
