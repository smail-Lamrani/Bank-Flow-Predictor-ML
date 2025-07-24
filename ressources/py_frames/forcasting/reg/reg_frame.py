import sys
import ast
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFrame, QVBoxLayout, QDialog, QMessageBox,
    QDialogButtonBox, QLabel, QPlainTextEdit, QListWidget, QListWidgetItem, QPushButton
)
from PyQt5.QtCore import Qt
import pdb
import joblib
import os

# Import your prebuilt ControlFrame
# ressources.py_frames.forcasting.
from ressources.py_frames.forcasting.reg.reg_view import ViewFrame

# Import data/model utilities and sklearn models
from ressources.py_frames.forcasting.reg.reg_utils import (
    get_data, prepare_and_split, train_model,
    evaluate_model, predict_model, plot_client_history,
    forecast_client
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Registry of available models
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "adaboost": AdaBoostRegressor,
    "ridge": Ridge,
    "lasso": Lasso,
    "knn": KNeighborsRegressor,
    "svr": SVR,
}

# Available channels for target selection
CHANNELS = [
    'nb_virements_adria_web_bo', 'nb_virements_agence_physique',
    'nb_virements_autre_digital', 'nb_virements_ebics', 'nb_virements_rpa',
    'mnt_virements_adria_web_bo', 'mnt_virements_agence_physique',
    'mnt_virements_autre_digital', 'mnt_virements_ebics', 'mnt_virements_rpa',
]

class TargetsDialog(QDialog):
    """Dialog for selecting channels with checkable list and select/deselect all."""
    def __init__(self, parent=None, selected=None):
        super().__init__(parent)
        self.setWindowTitle("Select Targets")
        self.resize(300, 400)
        layout = QVBoxLayout(self)

        self.list_w = QListWidget(self)
        for ch in CHANNELS:
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if selected and ch in selected else Qt.Unchecked)
            self.list_w.addItem(item)
        layout.addWidget(self.list_w)

        select_all = QPushButton("Select All")
        deselect_all = QPushButton("Deselect All")
        select_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        deselect_all.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        layout.addWidget(select_all)
        layout.addWidget(deselect_all)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.list_w.count()):
            self.list_w.item(i).setCheckState(state)

    def get_selected(self):
        return [self.list_w.item(i).text() for i in range(self.list_w.count())
                if self.list_w.item(i).checkState() == Qt.Checked]

class ModelConfigDialog(QDialog):
    """Dialog for entering model parameters as a Python dict."""
    def __init__(self, parent=None, initial_text="{}"):
        super().__init__(parent)
        self.setWindowTitle("Configure Model Parameters")
        self.resize(400, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter parameters as a Python dict:"))
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setPlainText(initial_text)
        layout.addWidget(self.text_edit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_params(self):
        text = self.text_edit.toPlainText()
        try:
            params = ast.literal_eval(text)
            if not isinstance(params, dict): raise ValueError("Input is not a dict.")
            return params
        except Exception as e:
            raise ValueError(f"Invalid dict: {e}")

class BuildRegManager(QFrame):
    """Manager frame connecting ControlFrame view and model logic."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = ViewFrame()
        layout.addWidget(self.view)
        self.view.select_model.clear()
        self.view.select_model.addItems(sorted(MODEL_REGISTRY.keys()))
        self.parent = parent
        self.model_params = {}
        self.selected_targets = []
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.split_date = None
        self.in_cols = None
        self.out_cols = None
        self.full_data = None
        self.horizon = None
        self.selected_model_name = None
        self.selectedmodel_cls = None
        self._connect_signals()

    def _connect_signals(self):
        v = self.view
        v.btn_train.clicked.connect(self.on_train)
        v.btn_evaluate.clicked.connect(self.on_evaluate)
        v.btn_plot.clicked.connect(self.on_plot)
        v.btn_train_p.clicked.connect(self.on_final_train)
        v.btn_save_p.clicked.connect(self.on_save_model)
        v.btn_model_config.clicked.connect(self.on_configure_model)
        v.btn_select_target.clicked.connect(self.on_select_targets)

    def on_select_targets(self):
        dialog = TargetsDialog(self, selected=self.selected_targets)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_targets = dialog.get_selected()
            # Update the QListView with selected targets
            from PyQt5.QtCore import QStringListModel
            model = QStringListModel(self.selected_targets)
            self.view.select_targets.setModel(model)
            self.log(f"Selected targets: {self.selected_targets}")

    def log(self, message: str):
        self.view.log_text.append(message)

    def on_train(self):
        v = self.view
        self.full_data, _ = get_data(self.parent.get_data(), num=v.train_on_t.value())
        if not self.selected_targets:
            self.log("Error: No targets selected for training.")
            QMessageBox.warning(self, "Training Error", "Please select targets before training.")
            return
        self.log("Starting training for evaluation...")
        self.log(f"Data loaded ({len(self.full_data)} rows). Targets: {self.selected_targets}")
        self.selected_model_name = self.view.select_model.currentText()
        # selectedmodel_cls = MODEL_REGISTRY.get(self.selected_model_name)
        self.horizon = v.horizon.value()

        if self.full_data is None or self.full_data.empty:
            self.log("Error: Data not loaded. Please ensure data is available before training for evaluation.")
            QMessageBox.critical(self, "Data Error", "Data not loaded. Cannot proceed with training.")
            return

        try:
            self.X_train, self.X_test, self.y_train, self.y_test, self.split_date, self.in_cols, self.out_cols = prepare_and_split(
                self.full_data, self.selected_targets, n_lags=v.n_lags.value(), horizon=self.horizon, test_months=v.test_months.value()
            )

            if self.selected_model_name:
                # self.selectedmodel_cls = self.selectedmodel_cls(**self.model_params)
                # training logic...
                self.pipeline = train_model(
                    self.X_train, self.y_train, self.in_cols,
                    model_name=self.selected_model_name,
                    model_params=None#self.model_params  # Use configured parameters
                )
                self.log(f"Training completed with {self.selected_model_name}.")
            else:
                self.log("No valid model selected for training.")
                QMessageBox.warning(self, "Model Selection Error", "Please select a model before training.")
        except Exception as e:
            self.log(f"Error during training process: {e}")
            QMessageBox.critical(self, "Training Process Error", f"An error occurred: {e}")

    def on_evaluate(self):
        if self.pipeline is not None:
            if self.X_test is not None and self.y_test is not None and self.out_cols is not None:
                try:
                    self.log("Evaluating model...")
                    eval_df = evaluate_model(self.pipeline, self.X_test, self.y_test, self.out_cols)
                    self.log("Evaluation done.")
                    self.log(eval_df.to_string())
                except Exception as e:
                    self.log(f"Error during evaluation: {e}")
                    QMessageBox.critical(self, "Evaluation Error", f"Could not evaluate model: {e}")
            else:
                self.log("Error: Test data or output columns not available for evaluation. Ensure training was successful.")
                QMessageBox.warning(self, "Evaluation Error", "Test data or configuration missing. Cannot evaluate.")
        else:
            self.log("Train the model first.")
            QMessageBox.warning(self, "Evaluation Error", "No model trained. Please train a model first.")

    def on_plot(self):
        v = self.view
        if not all([self.pipeline, self.X_test is not None, self.y_test is not None, 
                     self.full_data is not None, self.split_date, 
                     self.selected_targets, self.horizon is not None, self.out_cols is not None]):
            self.log("Error: Not all components are ready for plotting. Ensure model is trained and data is prepared.")
            QMessageBox.warning(self, "Plotting Error", "Cannot plot. Ensure model is trained and all data components are available.")
            return

        self.log("Plotting results...")
        try:
            client_id_str = v.client_id.text().strip()
            if not client_id_str:
                QMessageBox.warning(self, "Input Error", "Please enter a Client ID.")
                self.log("Plotting aborted: Client ID is missing.")
                return
            client_id = int(client_id_str)

            preds_df = predict_model(self.pipeline, self.X_test, self.out_cols)
            self.log(f"Predictions shape {preds_df.shape}, X_test shape {self.X_test.shape}, y_test shape {self.y_test.shape}")
            
            plot_client_history(
                client_id=client_id,
                df=self.full_data,
                split_date=self.split_date,
                X_test=self.X_test,
                y_test=self.y_test,
                y_pred_df=preds_df,
                channels=self.selected_targets,
                horizon=self.horizon
            )
            self.log("Plot complete.")
        except ValueError:
            self.log(f"Error: Invalid Client ID '{v.client_id.text().strip()}'. Must be an integer.")
            QMessageBox.critical(self, "Input Error", "Invalid Client ID. Please enter a numeric value.")
        except Exception as e:
            self.log(f"Error during plotting: {e}")
            QMessageBox.critical(self, "Plotting Error", f"Could not generate plot: {e}")

    def on_final_train(self):
        v = self.view
        self.full_data, _ = get_data(self.parent.get_data(), num=v.train_on_f.value())
        if not self.selected_targets:
            self.log("Error: No targets selected for training.")
            QMessageBox.warning(self, "Training Error", "Please select targets before training.")
            return
        self.log("Starting training on all data...")
        try:
            self.log(f"Data loaded ({len(self.full_data)} rows). Targets: {self.selected_targets}")
            self.selected_model_name = self.view.select_model.currentText()
            self.horizon = v.horizon.value() # Horizon from UI
            
            # For final training, X_test and y_test might be empty if test_months is None,
            # but prepare_and_split should still return X_train and y_train for the full dataset.
            self.X_train, self.X_test, self.y_train, self.y_test, self.split_date, self.in_cols, self.out_cols = prepare_and_split(
                self.full_data, self.selected_targets, n_lags=v.n_lags.value(), horizon=self.horizon, test_months=None
            )

            if self.selected_model_name:
                self.pipeline = train_model(
                    self.X_train, self.y_train, self.in_cols, # Train on all available X_train, y_train
                    model_name=self.selected_model_name,
                    model_params=self.model_params
                )
                self.log(f"Training on all data completed with {self.selected_model_name}.")
            else:
                self.log("No valid model selected for final training.")
                QMessageBox.warning(self, "Model Selection Error", "Please select a model before final training.")
        except Exception as e:
            self.log(f"Error during final training process: {e}")
            QMessageBox.critical(self, "Final Training Error", f"An error occurred: {e}")

    def on_save_model(self):
        v = self.view
        self.log("Attempting to save model...")
        if self.pipeline is None:
            self.log("Error: No model has been trained yet. Please train a model first.")
            QMessageBox.warning(self, "Save Error", "No model trained yet. Please train a final model first.")
            return

        if not self.selected_model_name:
            self.log("Error: Model name not set. Cannot determine filename to save.")
            QMessageBox.warning(self, "Save Error", "Model name not set. Cannot determine filename.")
            return
        
        try:
            # Ensure n_lags and horizon values are current from the UI for artifacts
            current_n_lags = self.view.n_lags.value()
            current_horizon = self.view.horizon.value() # This is the horizon from the "Build Final Model" section

            save_dir = f"models/reg/{self.selected_model_name}_{len(self.selected_targets)}_vars_h{current_horizon}_lags{current_n_lags}_trainedon_{v.train_on_f.value()}"
            os.makedirs(save_dir, exist_ok=True)

            pipeline_filename = f"pipeline_{self.selected_model_name}.joblib"
            artifacts_filename = f"pipeline_{self.selected_model_name}_artifacts.joblib"
            
            pipeline_path = os.path.join(save_dir, pipeline_filename)
            artifacts_path = os.path.join(save_dir, artifacts_filename)

            artifacts = {
                "model_name": self.selected_model_name,
                "model_params": self.model_params,
                "selected_targets": self.selected_targets,
                "in_cols": self.in_cols, # Should be from the final training
                "out_cols": self.out_cols, # Should be from the final training
                "horizon": current_horizon, 
                "n_lags": current_n_lags,
                "trained_on_samples": v.train_on_f.value() # Samples used for final training
            }

            joblib.dump(self.pipeline, pipeline_path)
            joblib.dump(artifacts, artifacts_path)
            self.log(f"Model saved to {pipeline_path}")
            self.log(f"Artifacts saved to {artifacts_path}")
            QMessageBox.information(self, "Save Successful", f"Model and artifacts saved to:\n{save_dir}")
        except OSError as e:
            self.log(f"Error creating directory or writing file: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not create directory or write file: {e}")
        except Exception as e: # Catch other potential errors from joblib or value access
            self.log(f"Error saving model: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save model: {e}")

    def on_configure_model(self):
        self.log("Opening config dialog...")
        dialog = ModelConfigDialog(self, initial_text=str(self.model_params))
        if dialog.exec_() == QDialog.Accepted:
            try:
                self.model_params = dialog.get_params()
                self.log(f"Params updated: {self.model_params}")
            except ValueError as e:
                QMessageBox.warning(self, "Parameter Error", f"Invalid parameters: {e}")
                self.log(f"Param error: {e}")
        else:
            self.log("Config canceled.")


    