import sys
import ast
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFrame, QVBoxLayout, QDialog, QMessageBox, QFileDialog, QTableView,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel
import pdb
import joblib
import os
from pathlib import Path


import pandas as pd

from ressources.py_frames.data_preparation.visualisation.vis_view import ViewFrame


class VisualisationManager(QFrame):
    """Manager frame connecting ControlFrame view and model logic."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = ViewFrame()
        layout.addWidget(self.view)
        self.parent = parent
        self._connect_signals()

    def _connect_signals(self):
            v = self.view
            
            

   
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("ControlManager with Targets")
    manager = VisualisationManager()
    main_win.setCentralWidget(manager)
    main_win.resize(800, 600)
    main_win.show()
    sys.exit(app.exec_())
    