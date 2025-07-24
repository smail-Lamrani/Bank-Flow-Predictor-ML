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
)


class ViewFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Styled panel
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # self._apply_stylesheet()

        # Root horizontal layout: left content panel, right log panel
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left content panel
        content_frame = QFrame()
        content_frame.setFrameShape(QFrame.StyledPanel)
        content_frame.setFrameShadow(QFrame.Raised)
        content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(content_frame)

        # Right log panel
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_frame.setFrameShadow(QFrame.Raised)
        log_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(log_frame)

        # Content layout
        main_layout = QVBoxLayout(content_frame)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("build model for regression of nb and mnt")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(title)

        # Model selection section
        main_layout.addWidget(self._hline())
        model_layout = QVBoxLayout()
        model_layout.setSpacing(5)
        model_layout.addWidget(self._hline())
        model_layout.addWidget(QLabel("Select model:"))
        self.select_model = QComboBox()
        self.select_model.addItem("Select Model")
        model_layout.addWidget(self.select_model)
        model_layout.addWidget(self._hline())
        self.btn_model_config = QPushButton("Model config")
        model_layout.addWidget(self.btn_model_config)
        main_layout.addLayout(model_layout)

        # Targets selection section
        main_layout.addWidget(self._hline())
        target_layout = QVBoxLayout()
        target_layout.setSpacing(5)
        target_layout.addWidget(self._hline())
        target_layout.addWidget(QLabel("Select targets:"))
        # Button to open target selection dialog
        self.btn_select_target = QPushButton("Select Target")
        target_layout.addWidget(self.btn_select_target)
        self.select_targets = QListView()
        target_layout.addWidget(self.select_targets)
        main_layout.addLayout(target_layout)

        # Evaluation title
        main_layout.addWidget(self._hline())
        eval_title = QLabel("Model Evaluation")
        eval_title.setAlignment(Qt.AlignCenter)
        eval_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(eval_title)
        main_layout.addWidget(self._hline())

        # Evaluation parameters grid
        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(QLabel("Train on :"), 0, 0)
        self.train_on_t = QSpinBox()
        self.train_on_t.setRange(-1, 999999)
        self.train_on_t.setValue(50)
        grid.addWidget(self.train_on_t, 0, 1)
        grid.addWidget(QLabel("clients"), 0, 2)

        grid.addWidget(QLabel("Look back"), 1, 0)
        self.n_lags = QSpinBox()
        self.n_lags.setMinimum(1)
        self.n_lags.setValue(3)
        grid.addWidget(self.n_lags, 1, 1)
        grid.addWidget(QLabel("months"), 1, 2)

        grid.addWidget(QLabel("Horizon :"), 2, 0)
        self.horizon = QSpinBox()
        self.horizon.setMinimum(1)
        grid.addWidget(self.horizon, 2, 1)
        grid.addWidget(QLabel("months"), 2, 2)

        grid.addWidget(QLabel("Test on"), 3, 0)
        self.test_months = QSpinBox()
        self.test_months.setMinimum(1)
        self.test_months.setValue(5)
        grid.addWidget(self.test_months, 3, 1)
        grid.addWidget(QLabel("months"), 3, 2)
        main_layout.addLayout(grid)

        # Evaluation action buttons
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_layout.addWidget(self.btn_train)

        self.btn_evaluate = QPushButton("Evaluate")
        self.btn_evaluate.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_layout.addWidget(self.btn_evaluate)
        main_layout.addLayout(btn_layout)

        # Plot for specific client layout
        plot_layout = QHBoxLayout()
        self.btn_plot = QPushButton("Plot for client : ")
        self.client_id = QLineEdit()
        self.client_id.setText("5200000007369")
        plot_layout.addWidget(self.btn_plot)
        plot_layout.addWidget(self.client_id)
        main_layout.addLayout(plot_layout)

        # Final model section
        main_layout.addWidget(self._hline())
        main_layout.addWidget(self._hline())
        final_title = QLabel("build final model")
        final_title.setAlignment(Qt.AlignCenter)
        final_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(final_title)
        main_layout.addWidget(self._hline())

        # Horizon selector
        grid_f = QGridLayout()
        grid_f.addWidget(QLabel("Train on :"), 0, 0)
        self.train_on_f = QSpinBox()
        self.train_on_f.setRange(-1, 999999)
        self.train_on_f.setValue(50)
        grid_f.addWidget(self.train_on_f, 0, 1)
        grid_f.addWidget(QLabel("clients"), 0, 2)
        main_layout.addLayout(grid_f)

        # Final action buttons
        final_btn_layout = QHBoxLayout()
        self.btn_train_p = QPushButton(
            "Train"
        )  # Consider more descriptive: btn_train_final
        self.btn_train_p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        final_btn_layout.addWidget(self.btn_train_p)

        self.btn_save_p = QPushButton(
            "Save"
        )  # Consider more descriptive: btn_save_final
        self.btn_save_p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        final_btn_layout.addWidget(self.btn_save_p)
        main_layout.addLayout(final_btn_layout)

        # Log area in right panel
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(10)
        log_title = QLabel("Log Output")
        log_title.setAlignment(Qt.AlignCenter)
        log_layout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_text)

    def _apply_stylesheet(self):
        style_sheet_path = os.path.join(os.path.dirname(__file__), "reg_style.css")
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
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("ControlFrame Test")
    control = ViewFrame()
    main_win.setCentralWidget(control)
    main_win.resize(800, 800)
    main_win.show()
    sys.exit(app.exec_())
