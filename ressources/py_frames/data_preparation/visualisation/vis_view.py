import sys
from PyQt5.QtWidgets import (
    QApplication, QFrame, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QTextEdit, QSizePolicy, QListWidget,
    QMainWindow, QLineEdit, QGroupBox
)
from PyQt5.QtCore import Qt
import os


class ViewFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._apply_stylesheet()

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(5)

        content_frame = QFrame()
        content_frame.setFrameShape(QFrame.StyledPanel)
        content_frame.setFrameShadow(QFrame.Raised)
        content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(content_frame, 3)

        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_frame.setFrameShadow(QFrame.Raised)
        log_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(log_frame, 2)

        main_layout = QVBoxLayout(content_frame)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        title = QLabel("\u2713 Visualisation: Comming soon")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setFixedHeight(30)
        main_layout.addWidget(title)

        main_layout.addWidget(self._hline())
      

    def _apply_stylesheet(self):
        style_sheet_path = os.path.join(os.path.dirname(__file__), "vis_style.css")
        try:
            with open(style_sheet_path, "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Warning: Stylesheet file not found at {style_sheet_path}. Using default styles.")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")


    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return line

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("Visualisation GUI")
    view = ViewFrame()
    main_win.setCentralWidget(view)
    main_win.resize(900, 800)
    main_win.show()
    sys.exit(app.exec_())