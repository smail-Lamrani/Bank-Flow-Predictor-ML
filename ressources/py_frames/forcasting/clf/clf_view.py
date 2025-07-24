import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)


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

        title = QLabel("\u2728 Build Activity Detection Model")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setFixedHeight(30)
        main_layout.addWidget(title)

        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_model_selection())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_evaluation_group())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_plot_group())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_final_model_group())

        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_title = QLabel("Log Output")
        log_title.setAlignment(Qt.AlignCenter)
        log_title.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

    def _apply_stylesheet(self):
        style_sheet_path = os.path.join(os.path.dirname(__file__), "clf_style.css")
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

    def _create_model_selection(self):
        group = QGroupBox("Model Selection")
        layout = QVBoxLayout()
        self.select_model = QComboBox()
        self.select_model.addItem("Select")
        layout.addWidget(QLabel("Select classifier:"))
        layout.addWidget(self.select_model)
        self.btn_model_config = QPushButton("Configure Model")
        layout.addWidget(self.btn_model_config)
        group.setLayout(layout)
        return group

    def _create_evaluation_group(self):
        group = QGroupBox("Classifier Evaluation")
        layout = QVBoxLayout()
        grid = QGridLayout()

        self.train_on_t = QSpinBox()
        self.train_on_t.setRange(-1, 999999)
        self.train_on_t.setValue(50)
        grid.addWidget(QLabel("Train on:"), 0, 0)
        grid.addWidget(self.train_on_t, 0, 1)
        grid.addWidget(QLabel("clients"), 0, 2)

        self.n_lags = QSpinBox()
        self.n_lags.setMinimum(1)
        self.n_lags.setValue(3)
        grid.addWidget(QLabel("Look back:"), 1, 0)
        grid.addWidget(self.n_lags, 1, 1)
        grid.addWidget(QLabel("months"), 1, 2)

        self.test_months = QSpinBox()
        self.test_months.setMinimum(1)
        self.test_months.setValue(5)
        grid.addWidget(QLabel("Test on:"), 3, 0)
        grid.addWidget(self.test_months, 3, 1)
        grid.addWidget(QLabel("months"), 3, 2)

        layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_evaluate = QPushButton("Evaluate")
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_evaluate)
        layout.addLayout(btn_layout)

        group.setLayout(layout)
        return group

    def _create_plot_group(self):
        group = QGroupBox("Client-specific Prediction")
        layout = QHBoxLayout()
        self.btn_plot = QPushButton("predict for client:")
        self.client_id = QLineEdit("5200000007369")
        layout.addWidget(self.btn_plot)
        layout.addWidget(self.client_id)
        group.setLayout(layout)
        return group

    def _create_final_model_group(self):
        group = QGroupBox("Final Model")
        layout = QVBoxLayout()

        grid = QGridLayout()
        self.train_on_f = QSpinBox()
        self.train_on_f.setRange(-1, 999999)
        self.train_on_f.setValue(50)
        grid.addWidget(QLabel("Train on:"), 0, 0)
        grid.addWidget(self.train_on_f, 0, 1)
        grid.addWidget(QLabel("clients"), 0, 2)
        layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        self.btn_train_p = QPushButton("Train Final Model")
        self.btn_save_p = QPushButton("Save Model")
        btn_layout.addWidget(self.btn_train_p)
        btn_layout.addWidget(self.btn_save_p)
        layout.addLayout(btn_layout)

        group.setLayout(layout)
        return group


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("Activity Detection GUI")
    view = ViewFrame()
    main_win.setCentralWidget(view)
    main_win.resize(900, 800)
    main_win.show()
    sys.exit(app.exec_())
