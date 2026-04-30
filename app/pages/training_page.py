from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QFrame, QSizePolicy,
)

_PROJECT_ROOT    = Path(__file__).parent.parent.parent
_CSV_PATH        = _PROJECT_ROOT / "src" / "data" / "gestures.csv"
_SEQUENCES_DIR   = _PROJECT_ROOT / "src" / "data" / "sequences"
_MODEL_PATH      = _PROJECT_ROOT / "src" / "models" / "classifier.pkl"

_MIN_LSTM_RECS   = 5
_MIN_LSTM_CLASSES = 2


# ── Data stat helpers (pure, main thread) ─────────────────────────────────────

def _single_stats() -> tuple[str, bool]:
    """Return (summary_text, can_train) for the single-gesture dataset."""
    if not _CSV_PATH.exists():
        return "No training data. Capture single-gesture samples first.", False
    try:
        counts: dict[str, int] = {}
        with open(_CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
        if not counts:
            return "CSV exists but contains no rows.", False
        n_cls = len(counts)
        total = sum(counts.values())
        lines = [f"{total} samples  ·  {n_cls} class{'es' if n_cls != 1 else ''}"]
        for lbl, cnt in sorted(counts.items()):
            lines.append(f"  {lbl}: {cnt}")
        return "\n".join(lines), n_cls >= 2 and total >= 10
    except Exception as exc:
        return f"Error reading data: {exc}", False


def _lstm_stats() -> tuple[str, bool]:
    """Return (summary_text, can_train) for the sequence dataset."""
    if not _SEQUENCES_DIR.exists():
        return "No sequence recordings. Capture sequences first.", False
    subdirs = [d for d in sorted(_SEQUENCES_DIR.iterdir()) if d.is_dir()]
    if not subdirs:
        return "No gesture classes found in sequences directory.", False
    counts = {d.name: len(list(d.glob("*.npy"))) for d in subdirs}
    n_cls  = len(counts)
    total  = sum(counts.values())
    min_r  = min(counts.values())
    lines  = [f"{total} recordings  ·  {n_cls} class{'es' if n_cls != 1 else ''}"]
    for name, cnt in sorted(counts.items()):
        ok = "✓" if cnt >= _MIN_LSTM_RECS else f"✗ need {_MIN_LSTM_RECS}"
        lines.append(f"  {name}: {cnt} rec{'s' if cnt != 1 else ''}  {ok}")
    return "\n".join(lines), n_cls >= _MIN_LSTM_CLASSES and min_r >= _MIN_LSTM_RECS


# ── Background worker ─────────────────────────────────────────────────────────

class _TrainWorker(QObject):
    log      = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, train_fn) -> None:
        super().__init__()
        self._fn = train_fn  # () -> dict; progress_cb already bound by caller

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Widget helpers ────────────────────────────────────────────────────────────

def _primary_btn(text: str, h: int = 36) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
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


def _log_widget() -> QPlainTextEdit:
    w = QPlainTextEdit()
    w.setReadOnly(True)
    w.setMinimumHeight(160)
    w.setStyleSheet("""
        QPlainTextEdit {
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 4px;
            color: #b0b0b0;
            font-family: monospace;
            font-size: 11px;
            padding: 6px;
        }
    """)
    return w


# ── Page ──────────────────────────────────────────────────────────────────────

class TrainingPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._single_thread: QThread | None = None
        self._single_worker: _TrainWorker | None = None
        self._lstm_thread:   QThread | None = None
        self._lstm_worker:   _TrainWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Train")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        hdr.addWidget(title)
        hdr.addStretch()
        root.addLayout(hdr)
        root.addSpacing(4)

        subtitle = QLabel(
            "Train classifiers directly from captured data. "
            "Both trainers run in the background — the UI remains responsive."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: #666666;")
        root.addWidget(subtitle)
        root.addSpacing(18)

        # ── Two-column body ───────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_single_panel(), stretch=1)
        body.addWidget(self._build_lstm_panel(),   stretch=1)
        root.addLayout(body, stretch=1)

        self._refresh_stats()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_stats()

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_single_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border: 1px solid #2d2d2d;
                border-radius: 6px;
            }
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        header_lbl = QLabel("Single Gesture Classifier")
        header_lbl.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #cccccc; border: none;"
        )
        lay.addWidget(header_lbl)

        desc = QLabel(
            "Trains RandomForest + MLP on gestures.csv; picks the "
            "best model by 5-fold CV. Output: classifier.pkl"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #666666; border: none;")
        lay.addWidget(desc)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d2d2d; border: none;")
        lay.addWidget(sep)

        self._single_stats_lbl = QLabel("")
        self._single_stats_lbl.setStyleSheet(
            "font-size: 11px; color: #888888; font-family: monospace; border: none;"
        )
        self._single_stats_lbl.setWordWrap(True)
        lay.addWidget(self._single_stats_lbl)

        self._single_train_btn = _primary_btn("Train Single Gesture Classifier")
        self._single_train_btn.clicked.connect(self._start_single_training)
        lay.addWidget(self._single_train_btn)

        self._single_log = _log_widget()
        lay.addWidget(self._single_log, stretch=1)

        return panel

    def _build_lstm_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border: 1px solid #2d2d2d;
                border-radius: 6px;
            }
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        header_lbl = QLabel("Sequence (LSTM) Classifier")
        header_lbl.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #cccccc; border: none;"
        )
        lay.addWidget(header_lbl)

        desc = QLabel(
            "Trains a single-layer LSTM on sequence recordings. "
            "Requires ≥ 2 classes with ≥ 5 recordings each. "
            "Output: sequence_classifier.pt + meta.pkl"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #666666; border: none;")
        lay.addWidget(desc)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d2d2d; border: none;")
        lay.addWidget(sep)

        self._lstm_stats_lbl = QLabel("")
        self._lstm_stats_lbl.setStyleSheet(
            "font-size: 11px; color: #888888; font-family: monospace; border: none;"
        )
        self._lstm_stats_lbl.setWordWrap(True)
        lay.addWidget(self._lstm_stats_lbl)

        self._lstm_train_btn = _primary_btn("Train LSTM Sequence Classifier")
        self._lstm_train_btn.clicked.connect(self._start_lstm_training)
        lay.addWidget(self._lstm_train_btn)

        self._lstm_log = _log_widget()
        lay.addWidget(self._lstm_log, stretch=1)

        return panel

    # ── Stats refresh ─────────────────────────────────────────────────────────

    def _refresh_stats(self) -> None:
        single_text, single_ok = _single_stats()
        self._single_stats_lbl.setText(single_text)
        if self._single_worker is None:
            self._single_train_btn.setEnabled(single_ok)

        lstm_text, lstm_ok = _lstm_stats()
        self._lstm_stats_lbl.setText(lstm_text)
        if self._lstm_worker is None:
            self._lstm_train_btn.setEnabled(lstm_ok)

    # ── Single training ───────────────────────────────────────────────────────

    def _start_single_training(self) -> None:
        self._single_train_btn.setEnabled(False)
        self._single_log.clear()
        self._append_single("Starting single-gesture training…")

        def _train():
            from src.training.trainer import run_training, validate_dataset
            validate_dataset(_CSV_PATH)
            return run_training(
                _CSV_PATH,
                _MODEL_PATH,
                progress_cb=lambda msg: self._single_log_sig.emit(msg),
            )

        # Use a lambda-based approach: emit signal from callback
        # Worker has no access to signals from here, so we proxy via a local QObject
        self._single_thread = QThread()
        self._single_worker = _TrainWorker(None)  # fn set below after signal wiring
        self._single_worker.log.connect(self._on_single_log)
        self._single_worker.finished.connect(self._on_single_done)
        self._single_worker.error.connect(self._on_single_error)
        self._single_worker.finished.connect(self._single_thread.quit)
        self._single_worker.error.connect(self._single_thread.quit)

        w = self._single_worker

        def _train_fn():
            from src.training.trainer import run_training, validate_dataset
            validate_dataset(_CSV_PATH)
            return run_training(_CSV_PATH, _MODEL_PATH, progress_cb=w.log.emit)

        self._single_worker._fn = _train_fn
        self._single_worker.moveToThread(self._single_thread)
        self._single_thread.started.connect(self._single_worker.run)
        self._single_thread.start()

    @Slot(str)
    def _on_single_log(self, msg: str) -> None:
        self._append_single(msg)

    @Slot(dict)
    def _on_single_done(self, result: dict) -> None:
        self._single_worker = None
        self._single_thread = None
        lines = [
            "─" * 40,
            f"  Model:     {result.get('model', '?')}",
            f"  Accuracy:  {result.get('accuracy', 0):.1%}",
            f"  CV Score:  {result.get('cv_score', 0):.1%}",
            "",
            "  Per-class recall:",
        ]
        for label, recall in sorted(result.get("per_class", {}).items()):
            lines.append(f"    {label:<14} {recall:.1%}")
        lines.append("─" * 40)
        self._append_single("\n".join(lines))
        self._refresh_stats()

    @Slot(str)
    def _on_single_error(self, msg: str) -> None:
        self._single_worker = None
        self._single_thread = None
        self._append_single(f"ERROR: {msg}")
        self._refresh_stats()

    # ── LSTM training ─────────────────────────────────────────────────────────

    def _start_lstm_training(self) -> None:
        self._lstm_train_btn.setEnabled(False)
        self._lstm_log.clear()
        self._append_lstm("Starting LSTM sequence training…")

        self._lstm_thread = QThread()
        self._lstm_worker = _TrainWorker(None)
        self._lstm_worker.log.connect(self._on_lstm_log)
        self._lstm_worker.finished.connect(self._on_lstm_done)
        self._lstm_worker.error.connect(self._on_lstm_error)
        self._lstm_worker.finished.connect(self._lstm_thread.quit)
        self._lstm_worker.error.connect(self._lstm_thread.quit)

        w = self._lstm_worker

        def _train_fn():
            from src.training.sequence_trainer import run_sequence_training
            return run_sequence_training(
                data_dir=_SEQUENCES_DIR,
                progress_cb=w.log.emit,
            )

        self._lstm_worker._fn = _train_fn
        self._lstm_worker.moveToThread(self._lstm_thread)
        self._lstm_thread.started.connect(self._lstm_worker.run)
        self._lstm_thread.start()

    @Slot(str)
    def _on_lstm_log(self, msg: str) -> None:
        self._append_lstm(msg)

    @Slot(dict)
    def _on_lstm_done(self, result: dict) -> None:
        self._lstm_worker = None
        self._lstm_thread = None
        lines = [
            "─" * 40,
            f"  Classes:        {', '.join(result.get('classes', []))}",
            f"  Accuracy:       {result.get('accuracy', 0):.1%}",
            f"  Val accuracy:   {result.get('val_accuracy', 0):.1%}",
            f"  Best epoch:     {result.get('epochs_trained', '?')}",
            f"  Sequence len:   {result.get('seq_len', '?')} frames",
            "",
            "  Recordings per class:",
        ]
        for name, cnt in sorted(result.get("recordings_per_class", {}).items()):
            lines.append(f"    {name:<14} {cnt}")
        lines.append("─" * 40)
        self._append_lstm("\n".join(lines))
        self._refresh_stats()

    @Slot(str)
    def _on_lstm_error(self, msg: str) -> None:
        self._lstm_worker = None
        self._lstm_thread = None
        self._append_lstm(f"ERROR: {msg}")
        self._refresh_stats()

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _append_single(self, msg: str) -> None:
        self._single_log.appendPlainText(msg)
        self._single_log.moveCursor(QTextCursor.MoveOperation.End)

    def _append_lstm(self, msg: str) -> None:
        self._lstm_log.appendPlainText(msg)
        self._lstm_log.moveCursor(QTextCursor.MoveOperation.End)
