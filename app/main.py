from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` and `app` are importable
# regardless of how the script is invoked.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from app.window import MainWindow
from app.config_manager import ConfigManager


def _apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(28, 28, 28))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(36, 36, 36))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 50))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(42, 42, 42))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText,      Qt.GlobalColor.red)
    p.setColor(QPalette.ColorRole.Link,            QColor(74, 158, 255))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(26, 74, 122))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(100, 100, 100))

    # Disabled state — slightly dimmer
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(90, 90, 90))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(90, 90, 90))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(90, 90, 90))
    app.setPalette(p)


def main() -> None:
    cfg = ConfigManager()
    errors = cfg.validate()
    if errors:
        # Print but don't abort — pages will show their own error states
        for e in errors:
            print(f"[config] WARNING: {e}", file=sys.stderr)

    app = QApplication(sys.argv)
    app.setApplicationName("Manus")
    app.setOrganizationName("Manus")
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
