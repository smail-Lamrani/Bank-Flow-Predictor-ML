import os
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QTableView,
    QPushButton,
    QToolButton,
    QComboBox,
    QSizePolicy,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QSpacerItem,
)
from PyQt5.QtCore import QCoreApplication, QMetaObject, Qt


class DataExplorationView(QFrame):
    def __init__(self, parent=None):
        super(DataExplorationView, self).__init__(parent)
        self.parent = parent
        self.setObjectName("DataExplorationView")
        self.resize(737, 552)
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_exploration_style.css")
        self.load_styles(css_path)  # Load external CSS
        self.setupUi()
 
    def load_styles(self, style_path):
        """Loads the QSS stylesheet from a file."""
        try:
            with open(style_path, "r") as f:
                style = f.read()
                self.setStyleSheet(style)
        except Exception as e:
            print("Error loading stylesheet:", e)

    def setupUi(self):
        # Create the main vertical layout
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(5, 5, 5, 5)
        mainLayout.setSpacing(5)

        # Create a top horizontal layout for controls
        topLayout = QHBoxLayout()
        topLayout.setSpacing(10)

        # Left group: Load Data and Combobox
        self.label_2 = QLabel(self)
        self.label_2.setObjectName("label_2")
        self.label_2.setText(
            QCoreApplication.translate("DataExplorationFrame", "Load Data", None)
        )

        self.btn_browse = QToolButton(self)
        self.btn_browse.setObjectName("btn_browse")
        self.btn_browse.setText(
            QCoreApplication.translate("DataExplorationFrame", "Load Data", None)
        )
        
        # New combobox added near the load button
        self.combo_view = QComboBox(self)
        self.combo_view.setObjectName("combo_view")
        self.combo_view.addItems(["Data", "Summary"])

        leftLayout = QHBoxLayout()
        leftLayout.addWidget(self.label_2)
        leftLayout.addWidget(self.btn_browse)
        leftLayout.addWidget(self.combo_view)

        # Right group: Clean Data
        self.label = QLabel(self)
        self.label.setObjectName("label")
        self.label.setText(
            QCoreApplication.translate("DataExplorationFrame", "Clean Data", None)
        )

        self.btn_clean_data = QPushButton(self)
        self.btn_clean_data.setObjectName("btn_clean_data")
        self.btn_clean_data.setText(
            QCoreApplication.translate("DataExplorationFrame", "Clean Data", None)
        )

        self.btn_fill_gaps = QPushButton(self)
        self.btn_fill_gaps.setObjectName("btn_fill_gaps")
        self.btn_fill_gaps.setText(
            QCoreApplication.translate("DataExplorationFrame", "fill gaps", None)
        )

        rightLayout = QHBoxLayout()
        rightLayout.addWidget(self.label)
        rightLayout.addWidget(self.btn_clean_data)
        rightLayout.addWidget(self.btn_fill_gaps) # Add the checkbox here

        # Add left and right groups into the top layout with a stretch between them
        topLayout.addLayout(leftLayout)
        topLayout.addStretch()
        topLayout.addLayout(rightLayout)

        # Wrap the top layout in a control panel widget for proper alignment.
        controlPanel = QFrame(self)
        controlPanel.setObjectName("controlPanel")
        controlPanel.setLayout(topLayout)

        # Add the control panel to the main layout with explicit top alignment.
        mainLayout.addWidget(controlPanel, 0, Qt.AlignTop)

        # Create and add the table view, which will expand as needed.
        self.t_show = QTableView(self)
        self.t_show.setObjectName("t_show")
        mainLayout.addWidget(self.t_show)

        # Set the main layout for the frame.
        self.setLayout(mainLayout)
        QMetaObject.connectSlotsByName(self)
