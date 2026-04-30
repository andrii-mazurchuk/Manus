from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QScrollArea, QFrame, QMessageBox, QSizePolicy,
)

from src.core.normalizer import normalize_landmarks_xyz, HAND_CONNECTIONS

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_SEQUENCES_DIR = _PROJECT_ROOT / "src" / "data" / "sequences"
_MODEL_PATH    = _PROJECT_ROOT / "src" / "models" / "hand_landmarker.task"
_MIN_FRAMES    = 10

# ── Shared widget styles ──────────────────────────────────────────────────────

_COMBO_STYLE = """
    QComboBox {
        background: #252525; border: 1px solid #383838; border-radius: 5px;
        color: #dddddd; font-size: 13px; padding: 0 10px;
    }
    QComboBox:focus { border-color: #4a9eff; }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background: #252525; color: #dddddd;
        selection-background-color: #1a4a7a;
        border: 1px solid #444444;
    }
"""

_SPIN_STYLE = """
    QSpinBox {
        background: #252525; border: 1px solid #383838; border-radius: 5px;
        color: #dddddd; font-size: 13px; padding: 0 6px;
    }
    QSpinBox:focus { border-color: #4a9eff; }
"""

_CAM_IDLE_STYLE = (
    "background: #111111; border: 1px solid #2d2d2d; "
    "border-radius: 6px; color: #444444; font-size: 14px;"
)
_CAM_REC_STYLE = (
    "background: #111111; border: 2px solid #cc2222; "
    "border-radius: 6px; color: #444444; font-size: 14px;"
)


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


def _danger_btn(text: str, h: int = 32) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    btn.setStyleSheet("""
        QPushButton {
            background: transparent; color: #cc4444;
            border: 1px solid #663333; border-radius: 5px;
            font-size: 12px; padding: 0 12px;
        }
        QPushButton:hover   { background: #3a1a1a; color: #ff5555; border-color: #cc4444; }
        QPushButton:pressed { background: #2a1010; }
    """)
    return btn


def _record_btn_active() -> str:
    return """
        QPushButton {
            background: #7a1a1a; color: #ffffff; border: none; border-radius: 5px;
            font-size: 14px; font-weight: 600; padding: 0 14px;
        }
        QPushButton:hover   { background: #9a2020; }
        QPushButton:pressed { background: #5a1010; }
    """


def _record_btn_idle() -> str:
    return """
        QPushButton {
            background: #1a4a7a; color: #ffffff; border: none; border-radius: 5px;
            font-size: 14px; font-weight: 600; padding: 0 14px;
        }
        QPushButton:hover   { background: #1f5a90; }
        QPushButton:pressed { background: #163d64; }
        QPushButton:disabled { background: #2a2a2a; color: #555555; }
    """


# ── Data helpers (pure functions, called from main thread) ────────────────────

def _save_sequence(name: str, frames: list[np.ndarray]) -> Path:
    label_dir = _SEQUENCES_DIR / name
    label_dir.mkdir(parents=True, exist_ok=True)
    arr  = np.stack(frames).astype(np.float32)
    ts   = int(time.time() * 1000)
    path = label_dir / f"{ts}.npy"
    np.save(path, arr)
    return path


def _sequence_counts() -> dict[str, int]:
    if not _SEQUENCES_DIR.exists():
        return {}
    return {
        d.name: len(list(d.glob("*.npy")))
        for d in sorted(_SEQUENCES_DIR.iterdir())
        if d.is_dir()
    }


def _clear_sequence_data(name: str) -> None:
    label_dir = _SEQUENCES_DIR / name
    if label_dir.exists():
        shutil.rmtree(label_dir)


# ── Camera worker (runs in QThread) ──────────────────────────────────────────

def _draw_hand(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 160, 160), 1, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, (0, 220, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)


class _CameraWorker(QObject):
    frame_ready    = Signal(bytes, object)  # (jpeg_bytes, np.ndarray[63] | None)
    error_occurred = Signal(str)
    finished       = Signal()

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._running = False

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
                "Run once: uv run src/data/extract_landmarks.py"
            )
            return

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        detector = mp_vision.HandLandmarker.create_from_options(options)
        cap      = cv2.VideoCapture(self._camera_index)
        ts_ms    = 0  # monotonically increasing, relative (not wall-clock)

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
                result = detector.detect_for_video(mp_img, ts_ms)
                ts_ms += 33

                flat = None
                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    _draw_hand(frame, lms)
                    flat = normalize_landmarks_xyz(lms)

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.frame_ready.emit(jpeg.tobytes(), flat)
        finally:
            cap.release()
            detector.close()


# ── Page ──────────────────────────────────────────────────────────────────────

class CaptureContinuousPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recording      = False
        self._frames: list[np.ndarray] = []
        self._record_start: float | None = None
        self._thread: QThread | None       = None
        self._worker: _CameraWorker | None = None

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._update_counter)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Capture — Sequence Gesture")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._hdr_status = QLabel("")
        self._hdr_status.setStyleSheet("font-size: 12px; color: #888888;")
        hdr.addWidget(self._hdr_status)
        root.addLayout(hdr)
        root.addSpacing(4)

        subtitle = QLabel(
            "Record temporal hand-gesture sequences for LSTM classifier training. "
            "Select or type a gesture name, start the camera, then use the record "
            "button to capture a sequence. Each recording saves as a (T, 63) .npy file."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: #666666;")
        root.addWidget(subtitle)
        root.addSpacing(14)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        ctrl.addWidget(_lbl("Gesture:"))
        self._gesture_combo = QComboBox()
        self._gesture_combo.setEditable(True)
        self._gesture_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._gesture_combo.setFixedHeight(34)
        self._gesture_combo.setMinimumWidth(180)
        self._gesture_combo.setStyleSheet(_COMBO_STYLE)
        ctrl.addWidget(self._gesture_combo)

        ctrl.addSpacing(12)
        ctrl.addWidget(_lbl("Camera:"))
        self._camera_spin = QSpinBox()
        self._camera_spin.setRange(0, 9)
        self._camera_spin.setValue(0)
        self._camera_spin.setFixedSize(64, 34)
        self._camera_spin.setStyleSheet(_SPIN_STYLE)
        ctrl.addWidget(self._camera_spin)

        ctrl.addSpacing(12)
        self._start_btn = _primary_btn("Start Camera")
        self._start_btn.clicked.connect(self._start_camera)
        ctrl.addWidget(self._start_btn)

        self._stop_cam_btn = _ghost_btn("Stop", w=70)
        self._stop_cam_btn.setEnabled(False)
        self._stop_cam_btn.clicked.connect(self._stop_camera)
        ctrl.addWidget(self._stop_cam_btn)

        ctrl.addStretch()
        root.addLayout(ctrl)
        root.addSpacing(14)

        # ── Body ──────────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)

        # Left: camera + record controls
        left = QVBoxLayout()
        left.setSpacing(8)

        self._cam_label = QLabel("Camera not started")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cam_label.setMinimumSize(480, 340)
        self._cam_label.setStyleSheet(_CAM_IDLE_STYLE)
        left.addWidget(self._cam_label, stretch=1)

        self._cam_status_lbl = QLabel("Camera not started")
        self._cam_status_lbl.setStyleSheet("font-size: 12px; color: #555555;")
        left.addWidget(self._cam_status_lbl)

        self._record_btn = QPushButton("⏺  Start Recording")
        self._record_btn.setFixedHeight(44)
        self._record_btn.setEnabled(False)
        self._record_btn.setStyleSheet(_record_btn_idle())
        self._record_btn.clicked.connect(self._toggle_recording)
        left.addWidget(self._record_btn)

        self._counter_lbl = QLabel("0 frames · 0.0 s")
        self._counter_lbl.setStyleSheet("font-size: 12px; color: #555555;")
        left.addWidget(self._counter_lbl)

        body.addLayout(left, stretch=1)

        # Right: stats panel
        body.addWidget(self._build_stats_panel())

        self._refresh_gesture_names()
        self._refresh_stats()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_gesture_names()
        self._refresh_stats()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._stop_camera()

    # ── Camera management ─────────────────────────────────────────────────────

    def _start_camera(self) -> None:
        self._stop_camera()

        self._thread = QThread()
        self._worker = _CameraWorker(self._camera_spin.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error_occurred.connect(self._on_camera_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

        self._start_btn.setEnabled(False)
        self._stop_cam_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        self._set_cam_status("Starting…", level=0)

    def _stop_camera(self) -> None:
        if self._recording:
            self._stop_recording()
        if self._worker:
            self._worker.request_stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None

        self._start_btn.setEnabled(True)
        self._stop_cam_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._cam_label.clear()
        self._cam_label.setText("Camera not started")
        self._cam_label.setStyleSheet(_CAM_IDLE_STYLE)
        self._set_cam_status("Camera stopped.", level=0)

    @Slot(bytes, object)
    def _on_frame_ready(self, jpeg: bytes, landmarks) -> None:
        img = QImage.fromData(jpeg)
        if not img.isNull():
            pix = QPixmap.fromImage(img).scaled(
                self._cam_label.width(),
                self._cam_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cam_label.setPixmap(pix)

        if self._recording and landmarks is not None:
            self._frames.append(landmarks)

        if landmarks is not None:
            if self._recording:
                self._set_cam_status("Recording — hand detected ✓", level=1)
            else:
                self._set_cam_status("Hand detected ✓", level=1)
        else:
            if self._recording:
                self._set_cam_status("No hand — frame skipped", level=-1)
            else:
                self._set_cam_status("No hand in frame", level=0)

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._stop_camera()
        self._set_cam_status(f"Error: {msg}", level=-1)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        name = self._gesture_combo.currentText().strip().upper()
        if not name:
            self._set_hdr_status("Enter a gesture name first.", ok=False)
            return
        if self._worker is None:
            self._set_hdr_status("Start the camera first.", ok=False)
            return

        self._recording     = True
        self._frames        = []
        self._record_start  = time.monotonic()

        self._record_btn.setText("⏹  Stop Recording")
        self._record_btn.setStyleSheet(_record_btn_active())
        self._cam_label.setStyleSheet(_CAM_REC_STYLE)
        self._elapsed_timer.start()
        self._update_counter()

    def _stop_recording(self) -> None:
        self._recording = False
        self._elapsed_timer.stop()

        self._record_btn.setText("⏺  Start Recording")
        self._record_btn.setStyleSheet(_record_btn_idle())
        if self._worker is not None:
            self._cam_label.setStyleSheet(_CAM_IDLE_STYLE)

        n = len(self._frames)
        self._counter_lbl.setText(f"{n} frames · {self._elapsed_seconds():.1f} s")

        if n < _MIN_FRAMES:
            self._set_hdr_status(
                f"Recording too short ({n} frames, need ≥ {_MIN_FRAMES}) — discarded.",
                ok=False,
            )
            self._frames = []
            return

        name = self._gesture_combo.currentText().strip().upper()
        try:
            path = _save_sequence(name, self._frames)
            self._refresh_gesture_names()
            self._refresh_stats()
            self._set_hdr_status(
                f"Saved {n} frames → {path.parent.name}/{path.name}", ok=True
            )
        except Exception as exc:
            self._set_hdr_status(f"Save failed: {exc}", ok=False)
        finally:
            self._frames = []

    def _elapsed_seconds(self) -> float:
        if self._record_start is None:
            return 0.0
        return time.monotonic() - self._record_start

    def _update_counter(self) -> None:
        n = len(self._frames)
        self._counter_lbl.setText(f"{n} frames · {self._elapsed_seconds():.1f} s")

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _build_stats_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 6px;"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        lbl = QLabel("Dataset Stats")
        lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #aaaaaa;")
        layout.addWidget(lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d2d2d;")
        layout.addWidget(sep)

        self._stats_inner = QWidget()
        self._stats_inner_layout = QVBoxLayout(self._stats_inner)
        self._stats_inner_layout.setContentsMargins(0, 0, 0, 0)
        self._stats_inner_layout.setSpacing(3)
        self._stats_inner_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._stats_inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #1e1e1e; width: 5px; border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: #444; border-radius: 2px; min-height: 16px;
            }
        """)
        layout.addWidget(scroll, stretch=1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #2d2d2d;")
        layout.addWidget(sep2)

        self._total_lbl = QLabel("Total: 0 recordings")
        self._total_lbl.setStyleSheet("font-size: 12px; color: #666666;")
        layout.addWidget(self._total_lbl)

        self._clear_btn = _danger_btn("Clear selected gesture")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.clicked.connect(self._clear_gesture)
        layout.addWidget(self._clear_btn)

        return panel

    # ── Data refreshes ────────────────────────────────────────────────────────

    def _refresh_gesture_names(self) -> None:
        current = self._gesture_combo.currentText()
        self._gesture_combo.blockSignals(True)
        self._gesture_combo.clear()
        names = sorted(_sequence_counts().keys())
        self._gesture_combo.addItems(names)
        if current in names:
            self._gesture_combo.setCurrentText(current)
        elif current:
            self._gesture_combo.setCurrentText(current)
        self._gesture_combo.blockSignals(False)

    def _refresh_stats(self) -> None:
        counts = _sequence_counts()
        il     = self._stats_inner_layout

        while il.count() > 1:
            item = il.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        selected = self._gesture_combo.currentText().strip().upper()
        for label, count in sorted(counts.items()):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 1, 0, 1)

            is_sel = (label == selected)
            name_c = "#ffffff" if is_sel else "#cccccc"
            cnt_c  = "#4a9eff" if is_sel else "#777777"

            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(
                f"font-size: 12px; color: {name_c}; "
                + ("font-weight: 600;" if is_sel else "")
            )
            row_l.addWidget(name_lbl)
            row_l.addStretch()

            cnt_lbl = QLabel(f"{count} rec{'s' if count != 1 else ''}")
            cnt_lbl.setStyleSheet(f"font-size: 12px; color: {cnt_c};")
            row_l.addWidget(cnt_lbl)

            il.insertWidget(il.count() - 1, row_w)

        total = sum(counts.values())
        self._total_lbl.setText(
            f"Total: {total} recording{'s' if total != 1 else ''}"
        )

    def _clear_gesture(self) -> None:
        name = self._gesture_combo.currentText().strip().upper()
        if not name:
            return
        counts = _sequence_counts()
        n      = counts.get(name, 0)
        if n == 0:
            self._set_hdr_status(f"No recordings for '{name}'.", ok=False)
            return
        reply = QMessageBox.question(
            self,
            "Clear gesture recordings",
            f"Delete all {n} recording(s) for '{name}'?\n\n"
            "This permanently removes the .npy files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            _clear_sequence_data(name)
            self._refresh_gesture_names()
            self._refresh_stats()
            self._set_hdr_status(f"Cleared '{name}'.", ok=True)

    # ── Status helpers ────────────────────────────────────────────────────────

    def _set_cam_status(self, msg: str, *, level: int) -> None:
        color = {1: "#50d090", 0: "#555555", -1: "#e05050"}.get(level, "#555555")
        self._cam_status_lbl.setText(msg)
        self._cam_status_lbl.setStyleSheet(f"font-size: 12px; color: {color};")

    def _set_hdr_status(self, msg: str, *, ok: bool) -> None:
        color = "#50d090" if ok else "#e05050"
        self._hdr_status.setText(msg)
        self._hdr_status.setStyleSheet(f"font-size: 12px; color: {color};")


# ── Small helpers ─────────────────────────────────────────────────────────────

def _lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 13px; color: #aaaaaa;")
    return lbl
