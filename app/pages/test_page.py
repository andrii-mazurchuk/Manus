from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class TestPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        title = QLabel("Test")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #4a9eff;")
        layout.addWidget(title)

        desc = QLabel("Live camera feed with real-time gesture classification")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 14px; color: #888888;")
        layout.addWidget(desc)

        badge = QLabel("Phase 6 — not yet implemented")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "font-size: 11px; color: #555555; "
            "border: 1px solid #333333; border-radius: 4px; "
            "padding: 4px 10px; margin-top: 12px;"
        )
        layout.addWidget(badge)
