from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QRadioButton, QButtonGroup,
    QScrollArea, QFrame, QMessageBox, QSizePolicy,
)

from app.config_manager import ConfigManager
from src.core.normalizer import normalize_landmarks, HAND_CONNECTIONS

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_GESTURES_DIR = _PROJECT_ROOT / "src" / "data" / "gestures"
_CSV_PATH     = _PROJECT_ROOT / "src" / "data" / "gestures.csv"
_MODEL_PATH   = _PROJECT_ROOT / "src" / "models" / "hand_landmarker.task"

# Must match the header the existing trainer expects
_CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

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


# ── Data helpers (pure functions, called from main thread) ────────────────────

def _save_sample(label: str, flat: np.ndarray) -> None:
    """Save one 42-float frame: .npy file + row appended to gestures.csv."""
    label_dir = _GESTURES_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    np.save(label_dir / f"{ts}.npy", flat)

    write_header = not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CSV_HEADER)
        writer.writerow([label] + flat.tolist())


def _sample_counts() -> dict[str, int]:
    """Count .npy files per gesture label directory."""
    if not _GESTURES_DIR.exists():
        return {}
    return {
        d.name: len(list(d.glob("*.npy")))
        for d in sorted(_GESTURES_DIR.iterdir())
        if d.is_dir()
    }


def _clear_label_data(label: str) -> None:
    """Delete .npy dir for label and remove its rows from gestures.csv."""
    label_dir = _GESTURES_DIR / label
    if label_dir.exists():
        shutil.rmtree(label_dir)

    if not _CSV_PATH.exists():
        return

    tmp = _CSV_PATH.with_suffix(".tmp")
    try:
        kept_rows: list[list] = []
        header: list | None = None
        with open(_CSV_PATH, "r", newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            header = next(reader, None)
            for row in reader:
                if row and row[0] != label:
                    kept_rows.append(row)

        if kept_rows:
            with open(tmp, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                if header:
                    writer.writerow(header)
                writer.writerows(kept_rows)
            tmp.replace(_CSV_PATH)
        else:
            _CSV_PATH.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    except Exception:
        tmp.unlink(missing_ok=True)


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
    frame_ready    = Signal(bytes, object)  # (jpeg_bytes, np.ndarray[42] | None)
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

        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        detector = mp_vision.HandLandmarker.create_from_options(options)
        cap      = cv2.VideoCapture(self._camera_index)

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

                flat = None
                if result.hand_landmarks:
                    lms  = result.hand_landmarks[0]
                    _draw_hand(frame, lms)
                    flat = normalize_landmarks(lms)

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.frame_ready.emit(jpeg.tobytes(), flat)
        finally:
            cap.release()
            detector.close()


# ── Page ──────────────────────────────────────────────────────────────────────

class CaptureSinglePage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg            = ConfigManager()
        self._last_landmarks = None        # np.ndarray(42,) | None — written from main thread
        self._thread: QThread | None       = None
        self._worker: _CameraWorker | None = None

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._do_capture)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Capture — Single Gesture")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._hdr_status = QLabel("")
        self._hdr_status.setStyleSheet("font-size: 12px; color: #888888;")
        hdr.addWidget(self._hdr_status)
        root.addLayout(hdr)
        root.addSpacing(4)

        subtitle = QLabel(
            "Select a gesture label, start the camera, then press Space or "
            "'Capture Frame' to save a sample.  Auto mode captures every N ms "
            "whenever a hand is present."
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
        self._gesture_combo.setFixedHeight(34)
        self._gesture_combo.setMinimumWidth(160)
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

        self._stop_btn = _ghost_btn("Stop", w=70)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_camera)
        ctrl.addWidget(self._stop_btn)

        ctrl.addStretch()

        # ── Mode (right of controls) ───────────────────────────────────────
        ctrl.addWidget(_lbl("Mode:"))

        self._snap_radio = _radio("Snapshot")
        self._snap_radio.setChecked(True)
        self._snap_radio.toggled.connect(self._on_mode_changed)
        ctrl.addWidget(self._snap_radio)

        self._auto_radio = _radio("Auto")
        self._auto_radio.toggled.connect(self._on_mode_changed)
        ctrl.addWidget(self._auto_radio)

        _grp = QButtonGroup(self)
        _grp.addButton(self._snap_radio)
        _grp.addButton(self._auto_radio)

        ctrl.addWidget(_lbl("every"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(50, 2000)
        self._interval_spin.setSingleStep(50)
        self._interval_spin.setValue(200)
        self._interval_spin.setSuffix(" ms")
        self._interval_spin.setFixedSize(110, 34)
        self._interval_spin.setEnabled(False)
        self._interval_spin.setStyleSheet(_SPIN_STYLE)
        ctrl.addWidget(self._interval_spin)

        root.addLayout(ctrl)
        root.addSpacing(14)

        # ── Body ──────────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)

        # Left: camera display + capture controls
        left = QVBoxLayout()
        left.setSpacing(8)

        self._cam_label = QLabel("Camera not started")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cam_label.setMinimumSize(480, 340)
        self._cam_label.setStyleSheet(
            "background: #111111; border: 1px solid #2d2d2d; "
            "border-radius: 6px; color: #444444; font-size: 14px;"
        )
        left.addWidget(self._cam_label, stretch=1)

        self._cam_status_lbl = QLabel("Camera not started")
        self._cam_status_lbl.setStyleSheet("font-size: 12px; color: #555555;")
        left.addWidget(self._cam_status_lbl)

        self._capture_btn = _primary_btn("Capture Frame  (Space)", h=40)
        self._capture_btn.setEnabled(False)
        self._capture_btn.clicked.connect(self._do_capture)
        left.addWidget(self._capture_btn)

        body.addLayout(left, stretch=1)

        # Right: stats panel
        body.addWidget(self._build_stats_panel())

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._refresh_gesture_names()
        self._refresh_stats()

    # ── Key handler ───────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._snap_radio.isChecked() and self._capture_btn.isEnabled():
                self._do_capture()
        super().keyPressEvent(event)

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

        # Scrollable label list
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

        self._total_lbl = QLabel("Total: 0 samples")
        self._total_lbl.setStyleSheet("font-size: 12px; color: #666666;")
        layout.addWidget(self._total_lbl)

        self._clear_btn = _danger_btn("Clear selected gesture")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.clicked.connect(self._clear_gesture)
        layout.addWidget(self._clear_btn)

        return panel

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
        self._stop_btn.setEnabled(True)
        self._capture_btn.setEnabled(True)
        self._set_cam_status("Starting…", level=0)

        if self._auto_radio.isChecked():
            self._auto_timer.start(self._interval_spin.value())

    def _stop_camera(self) -> None:
        self._auto_timer.stop()
        if self._worker:
            self._worker.request_stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None
        self._last_landmarks = None

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._capture_btn.setEnabled(False)
        self._cam_label.clear()
        self._cam_label.setText("Camera not started")
        self._set_cam_status("Camera stopped.", level=0)

    @Slot(bytes, object)
    def _on_frame_ready(self, jpeg: bytes, landmarks) -> None:
        self._last_landmarks = landmarks

        img = QImage.fromData(jpeg)
        if not img.isNull():
            pix = QPixmap.fromImage(img).scaled(
                self._cam_label.width(),
                self._cam_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cam_label.setPixmap(pix)

        if landmarks is not None:
            self._set_cam_status("Hand detected ✓", level=1)
        else:
            self._set_cam_status("No hand in frame", level=-1)

    @Slot(str)
    def _on_camera_error(self, msg: str) -> None:
        self._stop_camera()
        self._set_cam_status(f"Error: {msg}", level=-1)

    # ── Capture ───────────────────────────────────────────────────────────────

    def _do_capture(self) -> None:
        if self._last_landmarks is None:
            self._set_cam_status("No hand — hold steady first", level=-1)
            return
        label = self._gesture_combo.currentText()
        if not label:
            return
        _save_sample(label, self._last_landmarks)
        self._refresh_stats()
        self._flash_capture()

    def _flash_capture(self) -> None:
        self._cam_label.setStyleSheet(
            "background: #111111; border: 2px solid #00e076; "
            "border-radius: 6px; color: #444444; font-size: 14px;"
        )
        QTimer.singleShot(200, self._reset_cam_border)

    def _reset_cam_border(self) -> None:
        self._cam_label.setStyleSheet(
            "background: #111111; border: 1px solid #2d2d2d; "
            "border-radius: 6px; color: #444444; font-size: 14px;"
        )

    # ── Mode ──────────────────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        auto = self._auto_radio.isChecked()
        self._interval_spin.setEnabled(auto)
        self._capture_btn.setVisible(not auto)

        if auto:
            if self._worker is not None:
                self._auto_timer.start(self._interval_spin.value())
        else:
            self._auto_timer.stop()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _refresh_gesture_names(self) -> None:
        current = self._gesture_combo.currentText()
        self._gesture_combo.blockSignals(True)
        self._gesture_combo.clear()
        names = self._cfg.gesture_names()
        self._gesture_combo.addItems(names)
        if current in names:
            self._gesture_combo.setCurrentText(current)
        self._gesture_combo.blockSignals(False)

    def _refresh_stats(self) -> None:
        counts = _sample_counts()
        il     = self._stats_inner_layout

        # Remove all stat rows (leave the stretch at the end)
        while il.count() > 1:
            item = il.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        selected = self._gesture_combo.currentText()
        for label, count in sorted(counts.items()):
            row_w  = QWidget()
            row_l  = QHBoxLayout(row_w)
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

            cnt_lbl = QLabel(str(count))
            cnt_lbl.setStyleSheet(f"font-size: 12px; color: {cnt_c};")
            row_l.addWidget(cnt_lbl)

            il.insertWidget(il.count() - 1, row_w)

        total = sum(counts.values())
        self._total_lbl.setText(
            f"Total: {total} sample{'s' if total != 1 else ''}"
        )

    def _clear_gesture(self) -> None:
        label = self._gesture_combo.currentText()
        if not label:
            return
        counts = _sample_counts()
        n      = counts.get(label, 0)
        if n == 0:
            self._set_hdr_status(f"No samples for '{label}'.", ok=False)
            return
        reply = QMessageBox.question(
            self,
            "Clear gesture data",
            f"Delete all {n} sample(s) for '{label}'?\n\n"
            "This removes the .npy files and their rows in gestures.csv.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            _clear_label_data(label)
            self._refresh_stats()
            self._set_hdr_status(f"Cleared '{label}'.", ok=True)

    # ── Status helpers ────────────────────────────────────────────────────────

    def _set_cam_status(self, msg: str, *, level: int) -> None:
        # level:  1 = good (green), 0 = neutral, -1 = warning (red)
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


def _radio(text: str) -> QRadioButton:
    rb = QRadioButton(text)
    rb.setStyleSheet("color: #cccccc; font-size: 13px;")
    return rb
