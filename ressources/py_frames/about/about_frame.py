import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QFrame, QMainWindow, QVBoxLayout
# ressources.py_frames.about.
from ressources.py_frames.about.about_view import ViewFrame


class AboutFrame(QFrame):
    update_ext_log = pyqtSignal(str)

    def __init__(self, parent=None):
        super(AboutFrame, self).__init__(parent)
        self.parent = parent

        layout = QVBoxLayout()

        self.view = ViewFrame(self)
        layout.addWidget(self.view)


# Test harness
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     main_win = QMainWindow()
#     main_win.setWindowTitle("ControlFrame Test")
#     control = AboutFrame()
#     main_win.setCentralWidget(control)
#     main_win.resize(800, 800)
#     main_win.show()
#     sys.exit(app.exec_())
