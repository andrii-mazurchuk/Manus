from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QDoubleSpinBox, QLineEdit,
    QMessageBox, QHeaderView, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Qt

from app.config_manager import ConfigManager

_COL_NAME      = 0
_COL_THRESHOLD = 1
_COL_DELETE    = 2

_ROW_H         = 44
_THRESH_COL_W  = 150
_DELETE_COL_W  = 100


def _primary_btn(text: str, h: int = 32, w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    if w:
        btn.setFixedWidth(w)
    btn.setStyleSheet("""
        QPushButton {
            background: #1a4a7a;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            font-size: 13px;
            padding: 0 14px;
        }
        QPushButton:hover   { background: #1f5a90; }
        QPushButton:pressed { background: #163d64; }
        QPushButton:disabled { background: #2a2a2a; color: #555555; }
    """)
    return btn


def _danger_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(28)
    btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #cc4444;
            border: 1px solid #663333;
            border-radius: 4px;
            font-size: 12px;
            padding: 0 10px;
        }
        QPushButton:hover   { background: #3a1a1a; color: #ff5555; border-color: #cc4444; }
        QPushButton:pressed { background: #2a1010; }
    """)
    return btn


def _threshold_spin(value: float, on_change) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.50, 1.00)
    spin.setSingleStep(0.05)
    spin.setDecimals(2)
    spin.setValue(value)
    spin.setFixedHeight(30)
    spin.setStyleSheet("""
        QDoubleSpinBox {
            background: #252525;
            border: 1px solid #383838;
            border-radius: 4px;
            color: #dddddd;
            padding: 0 4px;
        }
        QDoubleSpinBox:focus { border-color: #4a9eff; }
    """)
    spin.valueChanged.connect(on_change)
    return spin


class GesturesPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg   = ConfigManager()
        self._dirty = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(12)

        title = QLabel("Gestures")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        header.addWidget(title)

        header.addStretch()

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        header.addWidget(self._status_lbl)

        self._save_btn = _primary_btn("Save", h=34, w=80)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        header.addWidget(self._save_btn)

        root.addLayout(header)
        root.addSpacing(4)

        subtitle = QLabel(
            "Define the gesture labels the classifier will recognise. "
            "Each label maps to one trained hand-shape class. "
            "Double-click a name cell to rename it."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: #666666;")
        root.addWidget(subtitle)
        root.addSpacing(18)

        # ── Table ─────────────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Gesture Name", "Min Confidence", ""])
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._table.setStyleSheet("""
            QTableWidget {
                background: #1e1e1e;
                alternate-background-color: #232323;
                border: 1px solid #2d2d2d;
                border-radius: 6px;
                gridline-color: #2d2d2d;
            }
            QTableWidget::item { padding: 4px 8px; color: #dddddd; }
            QTableWidget::item:selected {
                background: #1a3a5a;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #181818;
                color: #888888;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.5px;
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid #2d2d2d;
            }
        """)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(_COL_NAME,      QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_COL_THRESHOLD, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(_COL_DELETE,    QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(_COL_THRESHOLD, _THRESH_COL_W)
        self._table.setColumnWidth(_COL_DELETE,    _DELETE_COL_W)

        self._table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._table)
        root.addSpacing(20)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d2d2d;")
        root.addWidget(sep)
        root.addSpacing(14)

        # ── Add gesture form ──────────────────────────────────────────────────
        add_label = QLabel("Add gesture")
        add_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #aaaaaa;")
        root.addWidget(add_label)
        root.addSpacing(8)

        add_row = QHBoxLayout()
        add_row.setSpacing(10)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Gesture name (e.g. THUMBS_UP)")
        self._name_input.setFixedHeight(36)
        self._name_input.setStyleSheet("""
            QLineEdit {
                background: #252525;
                border: 1px solid #383838;
                border-radius: 5px;
                color: #dddddd;
                font-size: 13px;
                padding: 0 10px;
            }
            QLineEdit:focus { border-color: #4a9eff; }
            QLineEdit::placeholder { color: #555555; }
        """)
        self._name_input.returnPressed.connect(self._add_gesture)
        add_row.addWidget(self._name_input, stretch=1)

        self._add_threshold = _threshold_spin(0.70, lambda _: None)
        self._add_threshold.setFixedWidth(_THRESH_COL_W)
        add_row.addWidget(self._add_threshold)

        add_btn = _primary_btn("Add Gesture", h=36)
        add_btn.clicked.connect(self._add_gesture)
        add_row.addWidget(add_btn)

        root.addLayout(add_row)
        root.addSpacing(8)

        self._add_error_lbl = QLabel("")
        self._add_error_lbl.setStyleSheet("font-size: 12px; color: #e05050;")
        root.addWidget(self._add_error_lbl)

        root.addStretch()

        # ── Initial load ──────────────────────────────────────────────────────
        self._load()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._dirty:
            self._load()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        gestures = self._cfg.load_gesture_labels()
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for g in gestures:
            self._append_row(g["name"], float(g["threshold"]))
        self._table.blockSignals(False)
        self._set_dirty(False)

    # ── Table row management ──────────────────────────────────────────────────

    def _append_row(self, name: str, threshold: float) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, _ROW_H)

        # Col 0 — name (editable text)
        item = QTableWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, _COL_NAME, item)

        # Col 1 — threshold spinbox
        spin = _threshold_spin(threshold, lambda _: self._set_dirty(True))
        # Wrap in container to add padding
        spin_wrapper = _cell_widget(spin, h_pad=8)
        self._table.setCellWidget(row, _COL_THRESHOLD, spin_wrapper)

        # Col 2 — delete button
        del_btn = _danger_btn("Remove")
        del_btn.clicked.connect(self._on_delete_clicked)
        del_wrapper = _cell_widget(del_btn, h_pad=10)
        self._table.setCellWidget(row, _COL_DELETE, del_wrapper)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == _COL_NAME:
            # Auto-uppercase names for consistency
            text = item.text().strip().upper()
            self._table.blockSignals(True)
            item.setText(text)
            self._table.blockSignals(False)
            self._set_dirty(True)

    def _on_delete_clicked(self) -> None:
        btn = self.sender()
        row = self._row_for_delete_button(btn)
        if row < 0:
            return
        name_item = self._table.item(row, _COL_NAME)
        name = name_item.text() if name_item else f"row {row + 1}"
        reply = QMessageBox.question(
            self,
            "Remove gesture",
            f"Remove '{name}'?\n\nThis only removes the label definition. "
            "Existing training data for this gesture is not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._table.removeRow(row)
            self._set_dirty(True)

    def _row_for_delete_button(self, btn: QPushButton) -> int:
        """Find the table row whose delete-column wrapper contains `btn`."""
        for row in range(self._table.rowCount()):
            wrapper = self._table.cellWidget(row, _COL_DELETE)
            if wrapper and btn in wrapper.findChildren(QPushButton):
                return row
        return -1

    # ── Add gesture form ──────────────────────────────────────────────────────

    def _add_gesture(self) -> None:
        self._add_error_lbl.setText("")
        name = self._name_input.text().strip().upper()

        if not name:
            self._add_error_lbl.setText("Name cannot be empty.")
            return

        existing = [
            (self._table.item(r, _COL_NAME).text().upper()
             if self._table.item(r, _COL_NAME) else "")
            for r in range(self._table.rowCount())
        ]
        if name in existing:
            self._add_error_lbl.setText(f"'{name}' already exists.")
            return

        self._append_row(name, self._add_threshold.value())
        self._name_input.clear()
        self._name_input.setFocus()
        self._set_dirty(True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _collect(self) -> list[dict] | None:
        """Read current table state into a list of gesture dicts, or None on error."""
        gestures: list[dict] = []
        seen: set[str] = set()

        for row in range(self._table.rowCount()):
            item = self._table.item(row, _COL_NAME)
            name = item.text().strip().upper() if item else ""
            if not name:
                self._set_status(f"Row {row + 1}: gesture name cannot be empty.", ok=False)
                return None
            if name in seen:
                self._set_status(f"Duplicate name: '{name}'.", ok=False)
                return None
            seen.add(name)

            spin_wrapper = self._table.cellWidget(row, _COL_THRESHOLD)
            spin = spin_wrapper.findChild(QDoubleSpinBox) if spin_wrapper else None
            threshold = spin.value() if spin else 0.70

            gestures.append({"name": name, "threshold": round(threshold, 2)})

        return gestures

    def _save(self) -> None:
        gestures = self._collect()
        if gestures is None:
            return
        try:
            self._cfg.save_gesture_labels(gestures)
            self._set_dirty(False)
            self._set_status(f"Saved {len(gestures)} gesture(s).", ok=True)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", ok=False)

    # ── State helpers ─────────────────────────────────────────────────────────

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._save_btn.setEnabled(dirty)
        if dirty:
            self._status_lbl.setText("● Unsaved changes")
            self._status_lbl.setStyleSheet("font-size: 12px; color: #f0a020;")
        else:
            # Clear if currently showing unsaved indicator (not a result message)
            if "Unsaved" in self._status_lbl.text():
                self._status_lbl.setText("")

    def _set_status(self, msg: str, *, ok: bool) -> None:
        color = "#50d090" if ok else "#e05050"
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"font-size: 12px; color: {color};")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cell_widget(inner: QWidget, *, h_pad: int = 6) -> QWidget:
    """Wrap `inner` in a centred, padded container for use as a table cell widget."""
    wrapper = QWidget()
    layout  = QHBoxLayout(wrapper)
    layout.setContentsMargins(h_pad, 6, h_pad, 6)
    layout.setSpacing(0)
    layout.addWidget(inner)
    return wrapper
