from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.pages.gestures_page import GesturesPage
from app.pages.sequences_page import SequencesPage
from app.pages.capture_single_page import CaptureSinglePage
from app.pages.capture_continuous_page import CaptureContinuousPage
from app.pages.training_page import TrainingPage
from app.pages.test_page import TestPage

_SIDEBAR_WIDTH = 200

_NAV_ITEMS: list[tuple[str, str, type]] = [
    # (nav label, emoji-less icon char, PageClass)
    ("Gestures",           "◈", GesturesPage),
    ("Sequences",          "⟳", SequencesPage),
    ("Capture — Single",   "⊕", CaptureSinglePage),
    ("Capture — Sequence", "⏺", CaptureContinuousPage),
    ("Train",              "⚙", TrainingPage),
    ("Test",               "▶", TestPage),
]

_NAV_STYLE = """
    QPushButton {{
        text-align: left;
        padding: 0 12px 0 36px;
        border: none;
        border-radius: 6px;
        font-size: 13px;
        color: #999999;
        background: transparent;
    }}
    QPushButton:hover  {{ background: #252525; color: #dddddd; }}
    QPushButton:checked {{ background: #1a4a7a; color: #ffffff; font-weight: 600; }}
"""


class _NavButton(QPushButton):
    def __init__(self, label: str, parent=None) -> None:
        super().__init__(label, parent)
        self.setCheckable(True)
        self.setFixedHeight(42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(_NAV_STYLE)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Manus — Gesture Studio")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Sidebar ────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(_SIDEBAR_WIDTH)
        sidebar.setStyleSheet("background: #181818;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(10, 18, 10, 18)
        sb.setSpacing(2)

        app_title = QLabel("MANUS")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_title.setFixedHeight(36)
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        app_title.setFont(title_font)
        app_title.setStyleSheet("color: #4a9eff; margin-bottom: 16px;")
        sb.addWidget(app_title)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #2a2a2a; margin-bottom: 8px;")
        sb.addWidget(sep)

        self._stack = QStackedWidget()
        self._nav_buttons: list[_NavButton] = []

        for label, _icon, PageClass in _NAV_ITEMS:
            btn = _NavButton(label)
            btn.clicked.connect(self._make_nav(len(self._nav_buttons)))
            sb.addWidget(btn)
            self._nav_buttons.append(btn)
            self._stack.addWidget(PageClass())

        sb.addStretch()

        # Version tag at bottom of sidebar
        version = QLabel("v0.1 — prototype")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("font-size: 10px; color: #444444;")
        sb.addWidget(version)

        root_layout.addWidget(sidebar)

        # ── Vertical divider ───────────────────────────────────────────────
        divider = QWidget()
        divider.setFixedWidth(1)
        divider.setStyleSheet("background: #2a2a2a;")
        root_layout.addWidget(divider)

        root_layout.addWidget(self._stack)

        self._select(0)

    # ── Navigation ─────────────────────────────────────────────────────────

    def _make_nav(self, index: int):
        def _handler():
            self._select(index)
        return _handler

    def _select(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == index)
        self._stack.setCurrentIndex(index)
