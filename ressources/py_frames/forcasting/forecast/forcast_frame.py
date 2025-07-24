import sys
from pathlib import Path

import joblib
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QMainWindow,
    QMessageBox,
    QTableView,
    QVBoxLayout,
)

from ressources.py_frames.forcasting.forecast.clf_utils import (
    predict_next_channel_for_client,
    prepare_data,
)

# ressources.py_frames.forcasting.forecast.
from ressources.py_frames.forcasting.forecast.forcast_view import ViewFrame
from ressources.py_frames.forcasting.forecast.reg_utils import forecast_client, get_data

regressor_names = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
    "ridge",
    "lasso",
    "knn",
    "svr",
]
classifier_names = [
    "logistic",
    "random_forest",
    "extra_trees",
    "gb",
    "hist_gb",
    "adaboost",
    "svm",
    "knn",
    "nb",
    "mlp",
    "xgb",
]


class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView."""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe

    def rowCount(self, parent=None):
        return self._dataframe.shape[0]

    def columnCount(self, parent=None):
        return self._dataframe.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            # Format numbers to 2 decimal places if they are float
            value = self._dataframe.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])
        return None


class DataFrameDialog(QDialog):
    """A dialog to display a pandas DataFrame in a QTableView."""

    def __init__(self, dataframe: pd.DataFrame, title="Forecast Results", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window) 
        self.setMinimumSize(600, 400)  # Set a reasonable minimum size
        self.resize(800, 400)  
        layout = QVBoxLayout(self)

        self.tableView = QTableView()
        for col in dataframe.select_dtypes(include='number').columns:
            dataframe[col] = dataframe[col].astype(int)
        self.model = PandasModel(dataframe)
        self.tableView.setModel(self.model)

        # Styling the table - you can expand this or move to a QSS file
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setStyleSheet(
            """
            QTableView {
                gridline-color: #d0d0d0;
                font-size: 10pt;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #c0c0c0;
                font-size: 10pt;
                font-weight: bold;
            }
            QTableView::item {
                padding: 4px;
            }
        """
        )
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.resizeColumnsToContents()

        layout.addWidget(self.tableView)

        # Dialog buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        layout.addWidget(buttonBox)

        self.setLayout(layout)


class Forecaster_Manager(QFrame):
    """Manager frame connecting ControlFrame view and model logic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = ViewFrame()
        layout.addWidget(self.view)
        self.parent = parent
        self._connect_signals()
        self.clf_path = str(Path.cwd() / "models" / "clf")
        self.reg_path = str(Path.cwd() / "models" / "reg")

        self.pip_clf = None
        self.pip_clf_arts = None
        self.pip_reg = None
        self.pip_reg_arts = None

    def log(self, message: str):
        self.view.log_text.append(message)

    def list_files(self, root_path):
        self.log(f"Attempting to list files in directory: {root_path}")
        file_dict = {}
        try:
            root = Path(root_path)
            if not root.is_dir():
                self.log(
                    f"Error: Provided path '{root_path}' is not a directory or does not exist."
                )
                return file_dict  # Return empty dict if root_path is not a directory

            for sub in root.iterdir():
                if sub.is_dir():
                    self.log(f"Scanning subdirectory: {sub.name}")
                    for f in sub.iterdir():
                        if f.is_file():
                            # clé = nom du fichier, valeur = chemin absolu
                            file_dict[f.name] = str(f.resolve())
                            self.log(f"Found file: {f.name} in {sub.name}")
            self.log(
                f"Successfully listed {len(file_dict)} files from {root_path} and its subdirectories."
            )
        except PermissionError as e:
            self.log(f"Permission error while trying to list files in {root_path}: {e}")
        except Exception as e:
            self.log(
                f"An unexpected error occurred while listing files in {root_path}: {e}"
            )
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}")

        return file_dict

    def _connect_signals(self):
        v = self.view
        v.btn_load_clf.clicked.connect(self.on_load_clf)
        v.btn_load_reg.clicked.connect(self.on_load_reg)
        v.btn_detect_activity.clicked.connect(self.detect_activity)
        v.btn_forecast_regressor.clicked.connect(self.forecast_nb_mnt)

    def forecast_nb_mnt(self):
        v = self.view
        self.log("Attempting to forecast NB/MNT...")
        try:
            # --- 1. Check if regressor model and artifacts are loaded ---
            if self.pip_reg is None or self.pip_reg_arts is None:
                self.log(
                    "Pre-check failed: Regressor model or its artifacts are not loaded."
                )
                QMessageBox.warning(
                    self,
                    "Model Not Loaded",
                    "Please load the regressor model and its artifacts before forecasting.",
                )
                return

            reg = self.pip_reg
            arts = self.pip_reg_arts

            # --- 2. Validate Client ID ---
            client_id_text = v.client_id.text()
            if not client_id_text.isdigit():
                self.log(
                    f"Validation failed: Client ID '{client_id_text}' is not a digit."
                )
                QMessageBox.warning(
                    self, "Invalid Input", "Client ID must be a valid number."
                )
                return
            client_id = int(client_id_text)
            self.log(f"Client ID validated: {client_id}")

            df = self.parent.get_data()
            df, _ = get_data(
                df, num=-1
            )  # Assuming get_data can handle potential errors
            if df is None or df.empty:
                self.log(
                    "Data loading failed: get_data returned None or empty DataFrame after initial load."
                )
                QMessageBox.warning(
                    self,
                    "Data Error",
                    "Could not load data or the data is empty after processing.",
                )
                return
            df["date"] = pd.to_datetime(df["date"])
            self.log(f"Data loaded and preprocessed successfully.")

            # --- 4. Check for necessary keys in artifacts ---
            required_keys = ["selected_targets", "in_cols", "n_lags", "horizon"]
            for key in required_keys:
                if key not in arts:
                    self.log(
                        f"Artifact check failed: Missing key '{key}' in regressor artifacts."
                    )
                    QMessageBox.critical(
                        self,
                        "Artifact Error",
                        f"Missing key '{key}' in regressor model artifacts.",
                    )
                    return
            self.log("All required artifact keys found.")

            # --- 5. Perform Forecast ---
            self.log(
                f"Starting forecast for client ID: {client_id} with horizon {arts['horizon']}..."
            )
            fc = forecast_client(
                reg,
                df,
                client_id=client_id,
                channels=arts["selected_targets"],
                in_cols=arts["in_cols"],
                n_lags=arts["n_lags"],
                horizon=arts["horizon"],
            )
            self.log(f"Forecast completed for client {client_id}.")
            if fc is not None and not fc.empty:
                self.log("Forecast results (fc) will be shown in a dialog.")
                # Display fc in the new dialog
                dialog = DataFrameDialog(
                    fc, title=f"Forecast for Client {client_id}", parent=self
                )
                dialog.exec_()
            else:
                self.log("Forecast results (fc): No data or empty DataFrame returned.")
                QMessageBox.information(
                    self,
                    "Forecast Complete",
                    f"Forecast for client {client_id} generated no data.",
                )

        except (
            FileNotFoundError
        ) as e:  # Should be caught by explicit check, but good fallback
            self.log(f"File Error in forecast_nb_mnt: {e}")
            QMessageBox.critical(
                self, "File Error", f"An essential file was not found: {e.filename}"
            )
        except KeyError as e:
            self.log(
                f"Key Error in forecast_nb_mnt: Missing key {e} in model artifacts or data."
            )
            QMessageBox.critical(
                self,
                "Data/Artifact Error",
                f"Missing expected data or artifact key: {e}",
            )
        except (
            ValueError
        ) as e:  # For int(client_id_text) or other potential conversions
            self.log(f"Value Error in forecast_nb_mnt: {e}")
            QMessageBox.critical(
                self, "Input Error", f"There was an issue with an input value: {e}"
            )
        except Exception as e:
            self.log(f"An unexpected error occurred in forecast_nb_mnt: {e}")
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred during forecasting: {e}",
            )

    def detect_activity(self):
        v = self.view
        self.log("Attempting to detect activity...")
        try:
            # --- 1. Check if model and artifacts are loaded ---
            if self.pip_clf is None or self.pip_clf_arts is None:
                self.log("Pre-check failed: Classifier model or artifacts not loaded.")
                self.log(
                    "Error: Classifier model or its artifacts are not loaded. Please load them first."
                )
                QMessageBox.warning(
                    self,
                    "Model Not Loaded",
                    "Please load the classifier model and its artifacts before detecting activity.",
                )
                return

            # --- 2. Validate Client ID ---
            client_id_text = v.client_id.text()
            if not client_id_text.isdigit():
                self.log(
                    f"Validation failed: Client ID '{client_id_text}' is not a digit."
                )
                self.log(
                    f"Error: Invalid Client ID '{client_id_text}'. Must be a number."
                )
                QMessageBox.warning(
                    self, "Invalid Input", "Client ID must be a valid number."
                )
                return
            client_id = int(client_id_text)
            self.log(f"Client ID validated: {client_id}")

            # --- 3. Load and Prepare Data ---
            self.log("Loading and preparing data...")
            data = self.parent.get_data()
            if data.empty:
                QMessageBox.critical(self, "Data Empty", "Data empty")
                return
            self.log(f"Data confirmed")

            _, df = get_data(
                data, num=-1
            )  # Assuming get_data can handle potential errors or returns None/empty
            if df is None or df.empty:
                self.log(
                    "Data loading failed: get_data returned None or empty DataFrame."
                )
                self.log("Error: Failed to load or data is empty.")
                QMessageBox.warning(
                    self, "Data Error", "Could not load data or the data is empty."
                )
                return
            self.log("Data loaded successfully.")

            clf = self.pip_clf
            arts = self.pip_clf_arts
            lag_months = arts.get("n_lags")  # Use .get() for safer dictionary access
            if lag_months is None:
                self.log(
                    "Artifact check failed: 'n_lags' not found in model artifacts."
                )
                self.log("Error: 'n_lags' not found in model artifacts.")
                QMessageBox.critical(
                    self,
                    "Artifact Error",
                    "'n_lags' is missing from the model artifacts.",
                )
                return
            self.log(f"Model artifacts retrieved. Number of lags: {lag_months}")

            df_feat, features, le = prepare_data(
                df, lag_months=[val for val in range(1, lag_months + 1)]
            )
            self.log(f"Data prepared. Features for prediction: {features}")
            features = [
                va for va in features if va != "chan_t_adria_web_bo"
            ]  # Consider making this configurable or part of artifacts
            self.log(f"Features after filtering 'chan_t_adria_web_bo': {features}")

            # --- 4. Predict ---
            self.log(f"Starting prediction for client ID: {client_id}...")
            channel_probs = predict_next_channel_for_client(
                clf, df, features, client_id, [val for val in range(1, lag_months + 1)]
            )
            self.log(
                f"Predicted channel probabilities for client {client_id}: {channel_probs}"
            )

            # --- 5. Update UI ---
            self.log("Updating UI with prediction results...")
            v.adria_prob.setText(
                f"{100 * float(channel_probs.get('adria_web_bo', 0)):.2f} %"
            )
            v.agence_physique_prob.setText(
                f"{100 * float(channel_probs.get('agence_physique', 0)):.2f} %"
            )  # Changed default to 0 from '0'
            v.autre_digital_prob.setText(
                f"{100 * float(channel_probs.get('autre_digital', 0)):.2f} %"
            )  # Changed default to 0 from '0'
            v.inactif_prob.setText(
                f"{100 * float(channel_probs.get('inactif', 0)):.2f} %"
            )  # Changed default to 0 from '0'
            v.rpa_prob.setText(
                f"{100 * float(channel_probs.get('rpa', 0)):.2f} %"
            )  # Changed default to 0 from '0'
            self.log("UI updated successfully.")

        except FileNotFoundError as e:
            self.log(f"File Error in detect_activity: {e}")
            QMessageBox.critical(
                self, "File Error", f"An essential file was not found: {e.filename}"
            )
        except KeyError as e:
            self.log(
                f"Key Error in detect_activity: Missing key {e} in model artifacts or data."
            )
            QMessageBox.critical(
                self, "Data Error", f"Missing expected data or artifact key: {e}"
            )
        except ValueError as e:
            self.log(f"Value Error in detect_activity: {e}")
            QMessageBox.critical(
                self, "Input Error", f"There was an issue with an input value: {e}"
            )
        except Exception as e:  # General catch-all
            self.log(f"An unexpected error occurred in detect_activity: {e}")
            import traceback

            self.log(
                f"Traceback: {traceback.format_exc()}"
            )  # Log the full traceback for unexpected errors
            QMessageBox.critical(
                self, "Unexpected Error", f"An unexpected error occurred: {e}"
            )

    def on_load_clf(self):
        v = self.view
        self.log("Attempting to load classifier model...")
        try:
            paths, _ = QFileDialog.getOpenFileName(
                self,
                "Select .joblib classifier model file",
                self.clf_path,
                "Joblib files (*.joblib);;All files (*)",
            )

            if not paths:  # User cancelled the dialog
                self.log("Classifier model loading cancelled by user.")
                return
            if "artifacts" in paths:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    "Please select a valid .joblib model file that not contains the word 'artifacts'.",
                )
                return

            if paths.endswith(".joblib"):
                self.log(f"Selected classifier model file: {paths}")
                self.pip_clf = joblib.load(paths)
                self.log("Classifier model loaded successfully.")

                arts_path_str = paths.split(".")[0] + "_artifacts.joblib"
                arts_path = Path(arts_path_str)

                if not arts_path.exists():
                    self.log(
                        f"Error: Classifier artifacts file not found at {arts_path_str}"
                    )
                    QMessageBox.critical(
                        self,
                        "Artifacts Error",
                        f"Could not find classifier artifacts file: {arts_path.name}\nExpected at: {arts_path_str}",
                    )
                    self.pip_clf_arts = None  # Ensure it's reset
                    return

                self.pip_clf_arts = joblib.load(arts_path_str)
                self.log(
                    f"Classifier artifacts loaded successfully from {arts_path_str}."
                )
                self.log(
                    f"Loaded classifier artifacts: {self.pip_clf_arts}"
                )  # Replaces print
            else:
                self.log(f"Invalid file selected: {paths}. Not a .joblib file.")
                QMessageBox.warning(
                    self, "Invalid File", "Please select a valid .joblib model file."
                )

        except FileNotFoundError as e:
            self.log(f"File Error in on_load_clf: {e}")
            QMessageBox.critical(
                self,
                "File Load Error",
                f"Could not load file: {e.filename}\nMake sure the model and its artifacts file exist.",
            )
            self.pip_clf = None  # Reset on error
            self.pip_clf_arts = None
        except Exception as e:
            self.log(f"An unexpected error occurred in on_load_clf: {e}")
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"An unexpected error occurred while loading the classifier model: {e}",
            )
            self.pip_clf = None  # Reset on error
            self.pip_clf_arts = None

    def on_load_reg(self):
        v = self.view
        self.log("Attempting to load regressor model...")
        try:
            paths, _ = QFileDialog.getOpenFileName(
                self,
                "Select .joblib regressor model file",
                self.reg_path,
                "Joblib files (*.joblib);;All files (*)",
            )

            if not paths:  # User cancelled the dialog
                self.log("Regressor model loading cancelled by user.")
                return
            if "artifacts" in paths:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    "Please select a valid .joblib model file that not contains the word 'artifacts'.",
                )
                return

            if paths.endswith(".joblib"):
                self.log(f"Selected regressor model file: {paths}")
                self.pip_reg = joblib.load(paths)
                self.log("Regressor model loaded successfully.")

                arts_path_str = paths.split(".")[0] + "_artifacts.joblib"
                arts_path = Path(arts_path_str)

                if not arts_path.exists():
                    self.log(f"Error: Artifacts file not found at {arts_path_str}")
                    QMessageBox.critical(
                        self,
                        "Artifacts Error",
                        f"Could not find artifacts file: {arts_path.name}\nExpected at: {arts_path_str}",
                    )
                    self.pip_reg_arts = None  # Ensure it's reset
                    return

                self.pip_reg_arts = joblib.load(arts_path_str)
                self.log(
                    f"Regressor artifacts loaded successfully from {arts_path_str}."
                )

                v.regressor_results_list.clear()
                selected_targets = self.pip_reg_arts.get("selected_targets", [])
                v.regressor_results_list.addItems(selected_targets)
                self.log(
                    f"Regressor results list updated with targets: {selected_targets}"
                )
                self.log(
                    f"Loaded regressor artifacts: {self.pip_reg_arts}"
                )  # Replaces print
            else:
                self.log(f"Invalid file selected: {paths}. Not a .joblib file.")
                QMessageBox.warning(
                    self, "Invalid File", "Please select a valid .joblib model file."
                )

        except FileNotFoundError as e:
            self.log(f"File Error in on_load_reg: {e}")
            QMessageBox.critical(
                self,
                "File Load Error",
                f"Could not load file: {e.filename}\nMake sure the model and its artifacts file exist.",
            )
            self.pip_reg = None  # Reset on error
            self.pip_reg_arts = None
        except Exception as e:
            self.log(f"An unexpected error occurred in on_load_reg: {e}")
            import traceback

            self.log(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(
                self,
                "Load Error",
                f"An unexpected error occurred while loading the regressor model: {e}",
            )
            self.pip_reg = None  # Reset on error
            self.pip_reg_arts = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("ControlManager with Targets")
    manager = Forecaster_Manager()
    main_win.setCentralWidget(manager)
    main_win.resize(800, 600)
    main_win.show()
    sys.exit(app.exec_())
