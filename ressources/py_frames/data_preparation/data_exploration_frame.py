import os
import pickle

import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
)

import ressources.py_frames.data_preparation.helper as helper
from common import ProgressBarDialog, run_with_progress
from ressources.py_frames.data_preparation.data_exploration_view import (
    DataExplorationView,
)


class Model(QtCore.QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def headerData(self, section, orientation, role):  # Add this function
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._df.columns[section])
            if orientation == QtCore.Qt.Vertical:  # Vertical headers
                return str(self._df.index[section])
        return None

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])

    def rowCount(self, index):
        return len(self._df)

    def columnCount(self, index):
        return len(self._df.columns)


class Data_exploration(QFrame):
    update_ext_log = pyqtSignal(str)

    def __init__(self, parent):
        super(Data_exploration, self).__init__(parent)
        self.parent = parent

        # Initialiser les données
        self.data = None

        # Create the main vertical layout with no margins or extra spacing
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create an instance of DataExplorationView
        self.data_exploration_view = DataExplorationView(self)
        # Ensure that the view expands with the window
        self.data_exploration_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.data_exploration_view)

        self.setLayout(layout)

        # Hide the table view until data is loaded
        self.data_exploration_view.t_show.hide()

        # Appliquer le style global
        style_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "global_style.css",
        )
        try:
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

        self.progress_dialog = ProgressBarDialog(self.parent)

        # Connect buttons to their functions
        self.data_exploration_view.btn_browse.clicked.connect(self.load_data)
        self.data_exploration_view.btn_clean_data.clicked.connect(self.clean_data)
        self.data_exploration_view.combo_view.currentIndexChanged.connect(
            self.update_table
        )
        self.data_exploration_view.btn_fill_gaps.clicked.connect(self.fill_gaps)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data Files (*.csv *.xlsx *.pkl)"
        )
        if not file_path:
            return

        try:
            # Extract file extension robustly
            ext = os.path.splitext(file_path)[1][1:].lower()
            loaders = {"csv": pd.read_csv, "xlsx": pd.read_excel, "pkl": pd.read_pickle}

            # Validate extension
            if ext not in loaders:
                QMessageBox.warning(self, "Warning", f"Unsupported file format: {ext}")
                return
            if getattr(self, "thread", None) and callable(
                getattr(self.thread, "isRunning", None)
            ):
                if self.thread.isRunning():
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Please wait for the current operation to finish.",
                    )
                    return
            # thread = run_with_progress(
            # parent=self.parent,
            # func=loaders[ext],
            # kwargs={'filepath_or_buffer':file_path},
            # feedback=True,
            # message="Loading data …",
            # cancellable=True)
            self.thread = run_with_progress(
                self.parent,
                loaders[ext],
                file_path,
                feedback=False,
                message="Loading data …",
            )
            self.thread.result.connect(self.on_load_data)
            self.thread.error.connect(self.on_error)
            self.thread.finished.connect(self.on_finished)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    # thred
    def on_load_data(self, data_result):
        if data_result is not None and not data_result.empty:
            try:
                data_result.reset_index(drop=True, inplace=True)
                self.parent.set_data(data_result)
                self.update_table()  # Update table after data is set
                QMessageBox.information(self, "Success", "Data loaded successfully!")
            except Exception as e:
                self.parent.set_data(None)  # Clear data on processing error
                self.update_table()  # Reflect cleared data in table
                QMessageBox.critical(
                    self, "Processing Error", f"Error processing loaded data: {str(e)}"
                )
        else:
            self.parent.set_data(None)  # Clear data if loaded data is None or empty
            self.update_table()  # Reflect cleared data in table
            QMessageBox.warning(
                self, "Load Warning", "No data was loaded or the file was empty."
            )

    def on_finished(self):
        self.parent.status("Done")

    def on_error(self, e):
        QMessageBox.critical(self, "Error", f"{str(e)}")

    def update_table(self):
        if self.parent.get_data() is not None:

            if self.data_exploration_view.combo_view.currentText() == "Summary":
                summary_df = helper.get_summary_as_df(self.parent.get_data())
                self.data_exploration_view.t_show.setModel(Model(summary_df))
            else:
                self.data_exploration_view.t_show.setModel(
                    Model(self.parent.get_data())
                )
            self.data_exploration_view.t_show.show()

    def fill_gaps(self):
        data = self.parent.get_data()
        # Ensure data is loaded before filling gaps
        if data.empty:
            QMessageBox.warning(self, "Warning", "No dataset loaded to fill gaps!")
            return
        # Show a confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm filling gaps",
            "Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        if getattr(self, "thread", None) and callable(
            getattr(self.thread, "isRunning", None)
        ):
            if self.thread.isRunning():
                QMessageBox.warning(
                    self, "Warning", "Please wait for the current operation to finish."
                )
                return

        # thread = run_with_progress(
        # parent=self.parent,
        # func=helper.fill_gaps,
        # args=(data,),
        # kwargs={},
        # feedback=True,
        # message="Filling Gaps …",
        # cancellable=True)
        self.thread = run_with_progress(
            self.parent,
            helper.fill_gaps,
            data,
            feedback=True,
            message="Filling Gaps …",
        )
        self.thread.result.connect(self.on_filling_gaps_result)
        self.thread.error.connect(self.on_error)
        self.thread.finished.connect(self.on_finished)

    def on_filling_gaps_result(self, result):
        if result is not None and not result.empty:
            self.on_clean_data_result(result)

    def clean_data(self):
        data = self.parent.get_data()
        if data.empty:
            QMessageBox.warning(self, "Warning", "No dataset loaded to clean!")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Data Cleaning",
            "Do you want to proceed with cleaning the data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        if getattr(self, "thread", None) and callable(
            getattr(self.thread, "isRunning", None)
        ):
            if self.thread.isRunning():
                QMessageBox.warning(
                    self, "Warning", "Please wait for the current operation to finish."
                )
                return

        self.thread = run_with_progress(
            self.parent,
            helper.clean_data,
            data,
            feedback=True,
            message="Data Cleaning …",
        )
        self.thread.result.connect(self.on_clean_data_result)
        self.thread.error.connect(self.on_error)
        self.thread.finished.connect(self.on_finished)

    def on_clean_data_result(self, result):
        if result is not None and not result.empty:
            self.parent.set_data(result)
            self.update_table()
            self.save_data()

    def save_data(self):
        # Show a confirmation dialog
        selected_format, file_filter = self.ask_save_clean_data()
        if selected_format:
            # Open a file dialog for saving
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save File", "", file_filter, options=options
            )

            if not file_path:
                QMessageBox.information(
                    self, "Cancelled", "Save operation was cancelled."
                )
                return
            data = self.parent.get_data()
            if getattr(self, "thread", None) and callable(
                getattr(self.thread, "isRunning", None)
            ):
                if self.thread.isRunning():
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Please wait for the current operation to finish.",
                    )
                    return

            self.thread = run_with_progress(
                self.parent,
                self.on_save,
                data,
                file_path,
                selected_format,
                message="Saving Data …",
            )
            self.thread.error.connect(self.on_error)
            self.thread.finished.connect(self.on_finished)

    def on_save(self, data, file_path, selected_format):
        if selected_format == "pkl":
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        elif selected_format == "xlsx":
            data.to_excel(file_path, index=False)
        elif selected_format == "csv":
            data.to_csv(file_path, index=False)

    def ask_save_clean_data(self):
        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Cleaned Data")

        # Set up the layout
        layout = QVBoxLayout(dialog)

        # Add a message label
        label = QLabel("Do you need to save cleaned data?")
        layout.addWidget(label)

        # Add a combobox with options
        comboBox = QComboBox(dialog)
        comboBox.addItems(["pkl", "xlsx", "csv"])
        layout.addWidget(comboBox)

        # Add Yes and No buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        layout.addWidget(buttonBox)

        # Connect the buttons to accept or reject the dialog
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)

        # Execute the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            selected_format = comboBox.currentText()
            file_filter = None
            if selected_format == "pkl":
                file_filter = "Pickle Files (*.pkl)"
            elif selected_format == "xlsx":
                file_filter = "Excel Files (*.xlsx)"
            elif selected_format == "csv":
                file_filter = "CSV Files (*.csv)"
            return selected_format, file_filter

        return None, None
