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

        title = QLabel("\u2728 Forecasting")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setFixedHeight(30)
        main_layout.addWidget(title)

        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_model_selection())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._client_selection())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_clf_results_group())
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._create_regressor_results_group())
        

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
        style_sheet_path = os.path.join(os.path.dirname(__file__), "forecast_style.css")
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

    def _create_model_selection(self):
        group = QGroupBox("Models Selection")
        layout = QHBoxLayout()
        self.btn_load_clf = QPushButton("Load classifier")
        self.btn_load_reg = QPushButton("Load regressor")
        layout.addWidget(self.btn_load_clf)
        layout.addWidget(self.btn_load_reg)
        group.setLayout(layout)
        group.setFixedHeight(70)
        return group
    
    def _client_selection(self):
        group = QGroupBox("target client")
        layout_client = QHBoxLayout()
        self.client_id = QLineEdit("5200000007369")
        layout_client.addWidget(QLabel("Client ID : "))
        layout_client.addWidget(self.client_id)
        group.setLayout(layout_client)

        group.setFixedHeight(70)
        return group

    def _create_clf_results_group(self):
        group = QGroupBox("Classifier Results")
        layout = QVBoxLayout()
        grid = QGridLayout()

        grid.addWidget(QLabel("Agence Physique : "), 0, 0)
        self.agence_physique_prob = QLabel("NaN")
        grid.addWidget(self.agence_physique_prob, 0, 1)

        grid.addWidget(QLabel("RPA : "), 1, 0)
        self.rpa_prob = QLabel("NaN")
        grid.addWidget(self.rpa_prob, 1, 1)

        grid.addWidget(QLabel("Autre Digital : "), 2, 0)
        self.autre_digital_prob = QLabel("NaN")
        grid.addWidget(self.autre_digital_prob, 2, 1)

        grid.addWidget(QLabel("ADRIA : "), 3, 0)
        self.adria_prob = QLabel("NaN")
        grid.addWidget(self.adria_prob, 3, 1)

        grid.addWidget(QLabel("Inactif : "), 4, 0)
        self.inactif_prob = QLabel("NaN")
        grid.addWidget(self.inactif_prob, 4, 1)

        layout.addLayout(grid)
        self.btn_detect_activity = QPushButton("Detect Activity")
        layout.addWidget(self.btn_detect_activity)

        group.setLayout(layout)
        return group

    def _create_regressor_results_group(self):
        group = QGroupBox("Regressor Results:")
        layout = QVBoxLayout() # Changed to QVBoxLayout for vertical arrangement

        # Scrollable list for items
        self.regressor_results_list = QListWidget()
        self.regressor_results_list.setFixedHeight(100)
        self.regressor_results_list.setAlternatingRowColors(True) # Optional: for better readability
        layout.addWidget(self.regressor_results_list)

        # Forecast button
        self.btn_forecast_regressor = QPushButton("Forecast")
        layout.addWidget(self.btn_forecast_regressor)

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