import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ViewFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Styled panel
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._apply_stylesheet()

        # Create a layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter) # Align content to the center

        # Create the "Coming soon" label
        coming_soon_label = QLabel("Coming soon")
        coming_soon_label.setAlignment(Qt.AlignCenter) # Center the text within the label
        coming_soon_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #555;") # Optional styling

        layout.addWidget(coming_soon_label)

    def _apply_stylesheet(self):
        style_sheet_path = os.path.join(os.path.dirname(__file__), "about_style.css")
        try:
            with open(style_sheet_path, "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(
                f"Warning: Stylesheet file not found at {style_sheet_path}. Using default styles."
            )
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return line


# Test harness
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_win = QMainWindow()
#     main_win.setWindowTitle("ControlFrame Test")
#     control = ViewFrame()
#     main_win.setCentralWidget(control)
#     main_win.resize(800, 800)
#     main_win.show()
#     sys.exit(app.exec_())
