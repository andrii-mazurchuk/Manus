from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QSpinBox,
    QComboBox, QScrollArea, QFrame, QStackedWidget,
    QMessageBox, QAbstractItemView, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from app.config_manager import ConfigManager

_DEFAULT_GAP_MS   = 900
_DEFAULT_TOTAL_MS = 3000
_LEFT_W           = 230


# ── Shared button factories ───────────────────────────────────────────────────

def _primary_btn(text: str, h: int = 32, w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    if w:
        btn.setFixedWidth(w)
    btn.setStyleSheet("""
        QPushButton {
            background: #1a4a7a; color: #ffffff;
            border: none; border-radius: 5px;
            font-size: 13px; padding: 0 14px;
        }
        QPushButton:hover   { background: #1f5a90; }
        QPushButton:pressed { background: #163d64; }
        QPushButton:disabled { background: #2a2a2a; color: #555555; }
    """)
    return btn


def _ghost_btn(text: str, h: int = 30, w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(h)
    if w:
        btn.setFixedWidth(w)
    btn.setStyleSheet("""
        QPushButton {
            background: transparent; color: #aaaaaa;
            border: 1px solid #3a3a3a; border-radius: 4px;
            font-size: 12px; padding: 0 10px;
        }
        QPushButton:hover   { background: #2a2a2a; color: #dddddd; border-color: #555555; }
        QPushButton:pressed { background: #1e1e1e; }
        QPushButton:disabled { color: #444444; border-color: #2a2a2a; }
    """)
    return btn


def _icon_btn(text: str, size: int = 26) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedSize(size, size)
    btn.setStyleSheet("""
        QPushButton {
            background: transparent; color: #888888;
            border: 1px solid #333333; border-radius: 4px;
            font-size: 13px;
        }
        QPushButton:hover   { background: #2a2a2a; color: #dddddd; }
        QPushButton:pressed { background: #1e1e1e; }
        QPushButton:disabled { color: #333333; border-color: #252525; }
    """)
    return btn


def _input(placeholder: str = "", h: int = 34) -> QLineEdit:
    le = QLineEdit()
    le.setPlaceholderText(placeholder)
    le.setFixedHeight(h)
    le.setStyleSheet("""
        QLineEdit {
            background: #252525; border: 1px solid #383838;
            border-radius: 5px; color: #dddddd;
            font-size: 13px; padding: 0 10px;
        }
        QLineEdit:focus { border-color: #4a9eff; }
    """)
    return le


# ── Token row widget ──────────────────────────────────────────────────────────

class _TokenRow(QWidget):
    """One row in the pattern builder: [combo] [↑] [↓] [×]"""

    changed  = Signal()
    move_up  = Signal(object)   # emits self
    move_dn  = Signal(object)   # emits self
    removed  = Signal(object)   # emits self

    def __init__(self, gesture_names: list[str], selected: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.addItems(gesture_names)
        self._combo.setFixedHeight(30)
        self._combo.setStyleSheet("""
            QComboBox {
                background: #252525; border: 1px solid #383838;
                border-radius: 4px; color: #dddddd;
                font-size: 13px; padding: 0 8px;
            }
            QComboBox:focus { border-color: #4a9eff; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background: #252525; color: #dddddd;
                selection-background-color: #1a4a7a;
                border: 1px solid #444444;
            }
        """)
        if selected and selected in gesture_names:
            self._combo.setCurrentText(selected)
        self._combo.currentIndexChanged.connect(self.changed)
        layout.addWidget(self._combo, stretch=1)

        up_btn = _icon_btn("↑")
        up_btn.clicked.connect(lambda: self.move_up.emit(self))
        layout.addWidget(up_btn)

        dn_btn = _icon_btn("↓")
        dn_btn.clicked.connect(lambda: self.move_dn.emit(self))
        layout.addWidget(dn_btn)

        rm_btn = _icon_btn("×")
        rm_btn.setStyleSheet(rm_btn.styleSheet().replace("#888888", "#cc4444"))
        rm_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(rm_btn)

    def value(self) -> str:
        return self._combo.currentText()

    def set_value(self, name: str) -> None:
        self._combo.setCurrentText(name)

    def refresh_names(self, gesture_names: list[str]) -> None:
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(gesture_names)
        if current in gesture_names:
            self._combo.setCurrentText(current)
        self._combo.blockSignals(False)


# ── Sequences page ────────────────────────────────────────────────────────────

class SequencesPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cfg        = ConfigManager()
        self._sequences:  list[dict] = []
        self._sel_idx:    int        = -1
        self._dirty:      bool       = False
        self._token_rows: list[_TokenRow] = []
        self._updating    = False   # suppress signals during programmatic load

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Sequences")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #dddddd;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        hdr.addWidget(self._status_lbl)

        self._save_btn = _primary_btn("Save All", h=34, w=90)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        hdr.addWidget(self._save_btn)

        root.addLayout(hdr)
        root.addSpacing(4)

        subtitle = QLabel(
            "Define token-pattern sequences. A sequence fires when its gesture "
            "labels appear in order within the timing window."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: #666666;")
        root.addWidget(subtitle)
        root.addSpacing(18)

        # ── Two-panel body ────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(0)
        root.addLayout(body, stretch=1)

        body.addWidget(self._build_left_panel())

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("color: #2d2d2d;")
        body.addWidget(divider)

        body.addWidget(self._build_right_panel(), stretch=1)

        self._load()

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(_LEFT_W)
        panel.setStyleSheet("background: #1e1e1e;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        new_btn = _ghost_btn("New", w=60)
        new_btn.clicked.connect(self._new_sequence)
        toolbar.addWidget(new_btn)

        dup_btn = _ghost_btn("Duplicate", w=80)
        dup_btn.clicked.connect(self._duplicate_sequence)
        toolbar.addWidget(dup_btn)

        toolbar.addStretch()

        self._del_btn = _ghost_btn("Delete", w=65)
        self._del_btn.setEnabled(False)
        self._del_btn.setStyleSheet(self._del_btn.styleSheet()
            .replace("color: #aaaaaa", "color: #cc4444")
            .replace("color: #dddddd", "color: #ff5555"))
        self._del_btn.clicked.connect(self._delete_sequence)
        toolbar.addWidget(self._del_btn)

        layout.addLayout(toolbar)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: 1px solid #2d2d2d;
                border-radius: 5px;
                color: #cccccc;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #1a3a5a;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected { background: #252525; }
        """)
        self._list.currentRowChanged.connect(self._on_list_selection)
        layout.addWidget(self._list, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        self._right_stack = QStackedWidget()

        # ── [0] Empty placeholder ─────────────────────────────────────────────
        empty = QWidget()
        el = QVBoxLayout(empty)
        el.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph = QLabel("Select a sequence to edit\nor click  New  to create one")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("font-size: 14px; color: #444444; line-height: 1.6;")
        el.addWidget(ph)
        self._right_stack.addWidget(empty)

        # ── [1] Editor ────────────────────────────────────────────────────────
        editor = QWidget()
        el2 = QVBoxLayout(editor)
        el2.setContentsMargins(24, 20, 24, 20)
        el2.setSpacing(14)

        # Name
        name_row = QHBoxLayout()
        name_lbl = QLabel("Name")
        name_lbl.setFixedWidth(120)
        name_lbl.setStyleSheet("font-size: 12px; color: #888888; font-weight: 600;")
        name_row.addWidget(name_lbl)
        self._name_input = _input("e.g. double_stop")
        self._name_input.textChanged.connect(self._on_editor_changed)
        name_row.addWidget(self._name_input, stretch=1)
        el2.addLayout(name_row)

        # Pattern section
        pat_lbl = QLabel("Pattern")
        pat_lbl.setStyleSheet("font-size: 12px; color: #888888; font-weight: 600;")
        el2.addWidget(pat_lbl)

        # Scroll area for token rows
        self._tokens_container = QWidget()
        self._tokens_layout    = QVBoxLayout(self._tokens_container)
        self._tokens_layout.setContentsMargins(0, 0, 0, 0)
        self._tokens_layout.setSpacing(2)
        self._tokens_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._tokens_container)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(180)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: #1e1e1e; width: 6px; border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #444444; border-radius: 3px; min-height: 20px;
            }
        """)
        el2.addWidget(scroll)

        add_token_btn = _ghost_btn("+ Add Token", h=30)
        add_token_btn.clicked.connect(self._add_token_row)
        el2.addWidget(add_token_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Preview
        preview_row = QHBoxLayout()
        preview_lbl = QLabel("Preview")
        preview_lbl.setFixedWidth(120)
        preview_lbl.setStyleSheet("font-size: 12px; color: #888888; font-weight: 600;")
        preview_row.addWidget(preview_lbl)
        self._preview_lbl = QLabel("—")
        self._preview_lbl.setStyleSheet(
            "font-size: 13px; color: #4a9eff; font-family: monospace; "
            "background: #1a1a1a; border-radius: 4px; padding: 4px 10px;"
        )
        self._preview_lbl.setWordWrap(True)
        preview_row.addWidget(self._preview_lbl, stretch=1)
        el2.addLayout(preview_row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2d2d2d;")
        el2.addWidget(sep)

        # Timing
        timing_lbl = QLabel("Timing")
        timing_lbl.setStyleSheet("font-size: 12px; color: #888888; font-weight: 600;")
        el2.addWidget(timing_lbl)

        gap_row = QHBoxLayout()
        gap_lbl = QLabel("Max gap between tokens")
        gap_lbl.setFixedWidth(220)
        gap_lbl.setStyleSheet("font-size: 13px; color: #cccccc;")
        gap_row.addWidget(gap_lbl)
        self._gap_spin = self._timing_spin(100, 5000, _DEFAULT_GAP_MS, "ms")
        gap_row.addWidget(self._gap_spin)
        gap_row.addStretch()
        el2.addLayout(gap_row)

        total_row = QHBoxLayout()
        total_lbl = QLabel("Max total duration")
        total_lbl.setFixedWidth(220)
        total_lbl.setStyleSheet("font-size: 13px; color: #cccccc;")
        total_row.addWidget(total_lbl)
        self._total_spin = self._timing_spin(500, 15000, _DEFAULT_TOTAL_MS, "ms")
        total_row.addWidget(self._total_spin)
        total_row.addStretch()
        el2.addLayout(total_row)

        # Connect timing spinboxes once here; _on_editor_changed guards via _updating
        self._gap_spin.valueChanged.connect(self._on_editor_changed)
        self._total_spin.valueChanged.connect(self._on_editor_changed)

        # Action label
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #2d2d2d;")
        el2.addWidget(sep2)

        action_row = QHBoxLayout()
        act_lbl = QLabel("Action label")
        act_lbl.setFixedWidth(120)
        act_lbl.setStyleSheet("font-size: 12px; color: #888888; font-weight: 600;")
        action_row.addWidget(act_lbl)
        self._action_input = _input("Optional — what this sequence triggers (e.g. mute)")
        self._action_input.textChanged.connect(self._on_editor_changed)
        action_row.addWidget(self._action_input, stretch=1)
        el2.addLayout(action_row)

        el2.addStretch()
        self._right_stack.addWidget(editor)

        return self._right_stack

    @staticmethod
    def _timing_spin(lo: int, hi: int, default: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(100)
        spin.setValue(default)
        spin.setSuffix(f" {suffix}")
        spin.setFixedHeight(32)
        spin.setFixedWidth(120)
        spin.setStyleSheet("""
            QSpinBox {
                background: #252525; border: 1px solid #383838;
                border-radius: 4px; color: #dddddd;
                font-size: 13px; padding: 0 6px;
            }
            QSpinBox:focus { border-color: #4a9eff; }
        """)
        return spin

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._dirty:
            self._load()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = self._cfg.load_sequences()
        self._sequences = list(data.get("sequences", []))
        # Ensure every seq has timing keys (use config defaults if absent)
        thresholds = self._cfg.load_thresholds()
        sm = thresholds.get("sequence_model", {})
        global_gap   = int(sm.get("default_max_gap_ms",   _DEFAULT_GAP_MS))
        global_total = int(sm.get("default_max_total_ms", _DEFAULT_TOTAL_MS))
        for seq in self._sequences:
            seq.setdefault("max_gap_ms",   global_gap)
            seq.setdefault("max_total_ms", global_total)

        self._sel_idx = -1
        self._refresh_list()
        self._show_editor(False)
        self._set_dirty(False)

    def _gesture_names(self) -> list[str]:
        names = self._cfg.gesture_names()
        return names if names else ["(no gestures defined)"]

    # ── Left panel ────────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for seq in self._sequences:
            self._list.addItem(_list_text(seq))
        self._list.blockSignals(False)

    def _on_list_selection(self, idx: int) -> None:
        if self._updating:
            return
        # Flush current editor state before switching
        if self._sel_idx >= 0:
            self._flush_editor()
        self._sel_idx = idx
        self._del_btn.setEnabled(idx >= 0)
        if idx < 0 or idx >= len(self._sequences):
            self._show_editor(False)
        else:
            self._show_editor(True)
            self._load_editor(self._sequences[idx])

    def _new_sequence(self) -> None:
        if self._sel_idx >= 0:
            self._flush_editor()
        names = self._gesture_names()
        new_seq: dict = {
            "name":        f"sequence_{len(self._sequences) + 1}",
            "pattern":     [names[0]] if names else [],
            "max_gap_ms":  _DEFAULT_GAP_MS,
            "max_total_ms": _DEFAULT_TOTAL_MS,
            "action":      "",
        }
        self._sequences.append(new_seq)
        self._updating = True
        self._list.addItem(_list_text(new_seq))
        self._list.setCurrentRow(len(self._sequences) - 1)
        self._updating = False
        self._sel_idx  = len(self._sequences) - 1
        self._del_btn.setEnabled(True)
        self._show_editor(True)
        self._load_editor(new_seq)
        self._name_input.selectAll()
        self._name_input.setFocus()
        self._set_dirty(True)

    def _duplicate_sequence(self) -> None:
        if self._sel_idx < 0:
            return
        self._flush_editor()
        import copy
        dup = copy.deepcopy(self._sequences[self._sel_idx])
        dup["name"] = dup["name"] + "_copy"
        self._sequences.append(dup)
        self._updating = True
        self._list.addItem(_list_text(dup))
        self._list.setCurrentRow(len(self._sequences) - 1)
        self._updating = False
        self._sel_idx  = len(self._sequences) - 1
        self._show_editor(True)
        self._load_editor(dup)
        self._set_dirty(True)

    def _delete_sequence(self) -> None:
        if self._sel_idx < 0:
            return
        name = self._sequences[self._sel_idx].get("name", "this sequence")
        reply = QMessageBox.question(
            self, "Delete sequence",
            f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        del self._sequences[self._sel_idx]
        self._sel_idx = -1
        self._refresh_list()
        self._show_editor(False)
        self._del_btn.setEnabled(False)
        self._set_dirty(True)

    # ── Right panel (editor) ──────────────────────────────────────────────────

    def _show_editor(self, visible: bool) -> None:
        self._right_stack.setCurrentIndex(1 if visible else 0)

    def _load_editor(self, seq: dict) -> None:
        self._updating = True
        self._name_input.setText(seq.get("name", ""))
        self._gap_spin.setValue(int(seq.get("max_gap_ms",  _DEFAULT_GAP_MS)))
        self._total_spin.setValue(int(seq.get("max_total_ms", _DEFAULT_TOTAL_MS)))
        self._action_input.setText(seq.get("action", ""))

        self._clear_token_rows()
        names = self._gesture_names()
        for token in seq.get("pattern", []):
            self._insert_token_row(token, names)

        self._updating = False
        self._refresh_preview()

    def _flush_editor(self) -> None:
        """Write current editor values back into self._sequences[self._sel_idx]."""
        if self._sel_idx < 0 or self._sel_idx >= len(self._sequences):
            return
        seq = self._sequences[self._sel_idx]
        seq["name"]        = self._name_input.text().strip()
        seq["pattern"]     = [r.value() for r in self._token_rows]
        seq["max_gap_ms"]  = self._gap_spin.value()
        seq["max_total_ms"] = self._total_spin.value()
        action = self._action_input.text().strip()
        seq["action"] = action
        # Refresh list item text
        item = self._list.item(self._sel_idx)
        if item:
            item.setText(_list_text(seq))

    def _on_editor_changed(self) -> None:
        if self._updating:
            return
        self._refresh_preview()
        # Live-update list item name while typing
        if self._sel_idx >= 0:
            seq = self._sequences[self._sel_idx]
            seq["name"] = self._name_input.text().strip()
            item = self._list.item(self._sel_idx)
            if item:
                item.setText(_list_text(seq))
        self._set_dirty(True)

    # ── Token row management ──────────────────────────────────────────────────

    def _clear_token_rows(self) -> None:
        for row in self._token_rows:
            self._tokens_layout.removeWidget(row)
            row.deleteLater()
        self._token_rows.clear()

    def _insert_token_row(self, token: str, gesture_names: list[str] | None = None) -> _TokenRow:
        names = gesture_names or self._gesture_names()
        row   = _TokenRow(names, selected=token)
        row.changed.connect(self._on_token_changed)
        row.move_up.connect(self._move_token_up)
        row.move_dn.connect(self._move_token_dn)
        row.removed.connect(self._remove_token_row)
        # Insert before the stretch (always the last item in the layout)
        insert_pos = self._tokens_layout.count() - 1
        self._tokens_layout.insertWidget(insert_pos, row)
        self._token_rows.append(row)
        return row

    def _add_token_row(self) -> None:
        names = self._gesture_names()
        self._insert_token_row(names[0] if names else "")
        self._refresh_preview()
        self._set_dirty(True)

    def _remove_token_row(self, row: _TokenRow) -> None:
        if row in self._token_rows:
            self._token_rows.remove(row)
            self._tokens_layout.removeWidget(row)
            row.deleteLater()
            self._refresh_preview()
            self._set_dirty(True)

    def _move_token_up(self, row: _TokenRow) -> None:
        idx = self._token_rows.index(row) if row in self._token_rows else -1
        if idx <= 0:
            return
        # Swap values (simpler than swapping widgets)
        a, b = self._token_rows[idx - 1], self._token_rows[idx]
        va, vb = a.value(), b.value()
        a.set_value(vb)
        b.set_value(va)
        self._refresh_preview()
        self._set_dirty(True)

    def _move_token_dn(self, row: _TokenRow) -> None:
        idx = self._token_rows.index(row) if row in self._token_rows else -1
        if idx < 0 or idx >= len(self._token_rows) - 1:
            return
        a, b = self._token_rows[idx], self._token_rows[idx + 1]
        va, vb = a.value(), b.value()
        a.set_value(vb)
        b.set_value(va)
        self._refresh_preview()
        self._set_dirty(True)

    def _on_token_changed(self) -> None:
        if self._updating:
            return
        self._refresh_preview()
        self._set_dirty(True)

    def _refresh_preview(self) -> None:
        tokens = [r.value() for r in self._token_rows]
        if tokens:
            self._preview_lbl.setText("  →  ".join(tokens))
        else:
            self._preview_lbl.setText("—  (add tokens above)")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        # Flush current editor before saving
        if self._sel_idx >= 0:
            self._flush_editor()

        # Validate
        errors = _validate_sequences(self._sequences)
        if errors:
            self._set_status("Cannot save: " + errors[0], ok=False)
            return

        data = self._cfg.load_sequences()
        data["sequences"] = self._sequences
        try:
            self._cfg.save_sequences(data)
            self._set_dirty(False)
            self._set_status(f"Saved {len(self._sequences)} sequence(s).", ok=True)
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
            if "Unsaved" in self._status_lbl.text():
                self._status_lbl.setText("")

    def _set_status(self, msg: str, *, ok: bool) -> None:
        color = "#50d090" if ok else "#e05050"
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"font-size: 12px; color: {color};")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_text(seq: dict) -> str:
    name  = seq.get("name", "unnamed")
    count = len(seq.get("pattern", []))
    token_str = "  →  ".join(seq.get("pattern", [])) or "empty"
    return f"{name}  ({count} tokens)"


def _validate_sequences(sequences: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for i, seq in enumerate(sequences):
        name = seq.get("name", "").strip()
        if not name:
            errors.append(f"Sequence {i + 1} has no name.")
        elif name in seen:
            errors.append(f"Duplicate sequence name: '{name}'.")
        else:
            seen.add(name)
        if len(seq.get("pattern", [])) < 2:
            errors.append(f"'{name}': pattern must have at least 2 tokens.")
    return errors
