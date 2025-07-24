import ast
import os
import sys
import traceback

import joblib
import pandas as pd  # For prettier logging
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

import ressources.py_frames.forcasting.clf.clf_utils as utils
from ressources.py_frames.forcasting.clf.clf_utils import (
    predict_next_channel_for_client,
)

# Import your prebuilt ControlFrame
from ressources.py_frames.forcasting.clf.clf_view import ViewFrame

# Import data/model utilities and sklearn models
# (sklearn models are used within clf_utils)


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
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_params(self):
        text = self.text_edit.toPlainText()
        try:
            params = ast.literal_eval(text)
            if not isinstance(params, dict):
                raise ValueError("Input is not a dict.")
            return params
        except Exception as e:
            raise ValueError(f"Invalid dict: {e}")


class BuildClfManager(QFrame):
    """
    Manager frame connecting the ViewFrame (UI) with the model building
    and evaluation logic from clf_utils.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.view = ViewFrame()
        layout.addWidget(self.view)

        self.view.select_model.clear()
        self.view.select_model.addItems(["Select"] + sorted(utils.CLASSIFIERS.keys()))

        self.model_params = {}
        self.current_model_name_for_params = (
            None  # Tracks which model self.model_params is for
        )
        self.pipeline = None
        self.parent = parent
        self.full_data = None
        self.df_feat = None
        self.features = None
        self.le = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train = None
        self.test = None
        self.selected_model_name = None
        self._connect_signals()

    def _connect_signals(self):
        v = self.view
        v.btn_train.clicked.connect(self.on_train)
        v.btn_evaluate.clicked.connect(self.on_evaluate)
        v.btn_plot.clicked.connect(self.on_forecast)
        v.btn_train_p.clicked.connect(self.on_final_train)
        v.btn_save_p.clicked.connect(self.on_save_model)
        v.btn_model_config.clicked.connect(self.on_configure_model)

    def log(self, message: str):
        """Appends a message to the log text area in the UI."""
        self.view.log_text.append(message)

    def on_forecast(self):
        """Handles the forecast button click to predict for a client."""
        v = self.view
        if not self.pipeline:
            self.log("Error: Model not trained. Please train a model first.")
            QMessageBox.warning(
                self, "Forecast Error", "Model not trained. Please train a model first."
            )
            return

        if self.full_data is None or self.features is None:
            self.log(
                "Error: Data or features not available. Ensure training was successful."
            )
            QMessageBox.warning(
                self,
                "Forecast Error",
                "Data or feature list not available for forecasting.",
            )
            return

        client_id_str = v.client_id.text()
        try:
            client_id = int(client_id_str)
        except ValueError:
            self.log(f"Error: Invalid Client ID '{client_id_str}'. Must be an integer.")
            QMessageBox.critical(
                self,
                "Input Error",
                f"Invalid Client ID: '{client_id_str}'. Please enter a numeric ID.",
            )
            return

        try:
            self.log(
                f"\nForecasting for client ID: {client_id} (Model: {self.selected_model_name or 'N/A'})"
            )

            current_n_lags = v.n_lags.value()
            if current_n_lags <= 0:
                self.log(
                    "Error: Number of lags must be greater than 0 for forecasting."
                )
                QMessageBox.critical(
                    self,
                    "Configuration Error",
                    "Number of lags must be positive for forecasting.",
                )
                return
            lag_months = list(range(1, current_n_lags + 1))

            channel_probs = predict_next_channel_for_client(
                pipeline=self.pipeline,
                df=self.full_data,  # Original data loaded by get_data
                features=self.features,  # Feature list model was trained on
                client_id=client_id,
                lag_months=lag_months,
            )

            if not channel_probs:
                self.log("No probabilities returned by the forecast function.")
                return

            self.log("Predicted probabilities:")
            for key, value in channel_probs.items():
                self.log(f"  {key}: {value:.2f}")
            self.log("-" * 20)

        except ValueError as ve:  # Specific error from predict_next_channel_for_client
            self.log(f"Forecasting error for client {client_id}: {ve}")
            QMessageBox.warning(self, "Forecast Error", str(ve))
        except Exception as e:
            self.log(
                f"An unexpected error occurred during forecast for client {client_id}: {e}"
            )
            QMessageBox.critical(
                self, "Forecast Error", f"An unexpected error occurred: {e}"
            )

    def _perform_training_step(
        self, num_samples: int, test_split_value: int, training_type: str
    ):
        """Helper method to perform a training step (evaluation or final)."""
        v = self.view
        self.log(f"\nStarting training for {training_type}...")

        # 1. Load data
        try:
            _, self.full_data = utils.get_data(self.parent.get_data(), num=num_samples)
            if self.full_data is None or self.full_data.empty:
                QMessageBox.critical(
                    self, "Data Error", "Data not loaded or empty. Cannot proceed."
                )
                return False
            self.log(
                f"Data loaded ({len(self.full_data)} rows) using up to {num_samples} clients for {training_type}."
            )
        except FileNotFoundError:
            QMessageBox.critical(self, "Data Loading Error", f"")
            return False
        except Exception as e:
            self.log(f"Error loading data: {e}")
            tb = traceback.format_exc()
            self.log("Full traceback:\n" + tb)
            QMessageBox.critical(
                self, "Data Loading Error", f"An error occurred while loading data: {e}"
            )
            return False

        # 2. Get model and parameters
        self.selected_model_name = self.view.select_model.currentText()
        if not self.selected_model_name or self.selected_model_name == "Select":
            self.log("No valid model selected for training.")
            QMessageBox.warning(
                self, "Model Selection Error", "Please select a model before training."
            )
            return False

        # Ensure model_params are set for the current model
        if (
            self.current_model_name_for_params != self.selected_model_name
            or not self.model_params
        ):
            self.model_params = getattr(
                utils.BestParams(), self.selected_model_name, {}
            )
            self.current_model_name_for_params = (
                self.selected_model_name
            )  # Update tracker
            self.log(
                f"Using default parameters for {self.selected_model_name}: {self.model_params}"
            )
        else:
            self.log(
                f"Using configured parameters for {self.selected_model_name}: {self.model_params}"
            )

        # 3. Prepare data and train
        try:
            current_n_lags = v.n_lags.value()
            if current_n_lags <= 0:
                self.log("Error: Number of lags must be greater than 0.")
                QMessageBox.critical(
                    self, "Configuration Error", "Number of lags must be positive."
                )
                return False
            lag_months = list(range(1, current_n_lags + 1))

            self.log(f"Preparing data with {current_n_lags} lag(s): {lag_months}")
            self.df_feat, self.features, self.le = utils.prepare_data(
                self.full_data, lag_months=lag_months
            )
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.train,
                self.test,  # train/test are full dataframes for inspection
            ) = utils.split_data(self.df_feat, self.features, test_split_value)

            if self.X_train is None or self.y_train is None or self.X_train.empty:
                self.log(
                    "Error: Training data is empty after split. Check data and split parameters."
                )
                QMessageBox.critical(
                    self, "Data Split Error", "Training data is empty after split."
                )
                return False
            if test_split_value > 0 and (
                self.X_test is None or self.y_test is None or self.X_test.empty
            ):
                self.log(
                    f"Warning: Test data is empty after split with test_split_value={test_split_value}. Evaluation might not be meaningful."
                )
                # QMessageBox.warning(self, "Data Split Warning", "Test data is empty. Evaluation might not be meaningful.")
                # Continue training, but evaluation will be skipped or fail.

            self.log(f"Training model {self.selected_model_name}...")
            self.pipeline, train_duration = utils.train_classifier(
                self.X_train, self.y_train, self.selected_model_name, self.model_params
            )
            self.log(
                f"Training for {training_type} completed with {self.selected_model_name} in {train_duration:.2f} mins."
            )
            return True

        except Exception as e:
            self.log(f"Error during {training_type} training process: {e}")
            QMessageBox.critical(
                self,
                f"{training_type.capitalize()} Training Error",
                f"An error occurred: {e}",
            )
            self.pipeline = None  # Ensure pipeline is reset on error
            self.X_train, self.y_train, self.X_test, self.y_test = (
                None,
                None,
                None,
                None,
            )
            self.features = None
            return False

    def on_train(self):
        """Handles the 'Train' button click for model evaluation."""
        v = self.view
        num_samples = v.train_on_t.value()
        test_split_value = v.test_months.value()
        self._perform_training_step(num_samples, test_split_value, "evaluation")

    def on_evaluate(self):
        """Handles the 'Evaluate' button click."""
        if not self.pipeline:
            self.log(
                "Error: Model not trained. Please train a model before evaluating."
            )
            QMessageBox.warning(
                self,
                "Evaluation Error",
                "No model trained. Please train a model first.",
            )
            return

        if (
            self.X_test is None
            or self.y_test is None
            or self.le is None
            or self.X_test.empty
        ):
            self.log(
                "Error: Test data, labels, or label encoder not available or empty. "
                "Ensure training (with a non-zero test split) was successful."
            )
            QMessageBox.warning(
                self,
                "Evaluation Error",
                "Test data or necessary components missing/empty. "
                "Cannot evaluate. Please run training with a valid test split.",
            )
            return

        try:
            self.log("Evaluating model...")
            classification_rep, metrics, cm, inference_duration = (
                utils.evaluate_and_plot(
                    self.pipeline, self.X_test, self.y_test, self.le
                )
            )
            self.log(f"Evaluation completed in: {inference_duration:.5f} min")
            try:
                # Log classification report in a more readable format
                report_df = pd.DataFrame(classification_rep).transpose()
                self.log("Classification Report:\n" + report_df.to_string())
            except Exception as report_ex:
                self.log(
                    f"Could not format classification report: {report_ex}\nRaw report: {classification_rep}"
                )

            self.log("Metrics:\n" + str(metrics))
            # cm is a numpy array, convert to string for simple logging
            self.log("Confusion Matrix (Actual x Predicted):\n" + str(cm))
            self.log("-" * 20)
            # The plot is shown by evaluate_and_plot itself.
        except Exception as e:
            self.log(f"Error during evaluation: {e}")
            QMessageBox.critical(
                self, "Evaluation Error", f"Could not evaluate model: {e}"
            )

    def on_final_train(self):
        """Handles the 'Train Final Model' button click."""
        v = self.view
        num_samples = v.train_on_f.value()
        # For final training, test_split_value is 0 (use all data for training)
        self._perform_training_step(num_samples, 0, "final production")

    def on_save_model(self):
        """Handles the 'Save Model' button click."""
        v = self.view
        self.log("Attempting to save model...")
        if self.pipeline is None:
            self.log(
                "Error: No model has been trained yet. Please train a model first."
            )
            QMessageBox.warning(
                self,
                "Save Error",
                "No model trained yet. Please train a final model first.",
            )
            return

        if not self.selected_model_name:
            self.log("Error: Model name not set. Cannot determine filename to save.")
            QMessageBox.warning(
                self, "Save Error", "Model name not set. Cannot determine filename."
            )
            return

        try:
            current_n_lags = self.view.n_lags.value()
            final_train_samples = self.view.train_on_f.value()

            if current_n_lags <= 0:
                self.log(
                    f"Error: Invalid number of lags ({current_n_lags}) for saving artifacts."
                )
                QMessageBox.critical(
                    self,
                    "Save Error",
                    "Number of lags must be positive to save model artifacts.",
                )
                return

            save_dir = f"models/clf/{self.selected_model_name}_lags{current_n_lags}_trainedon_{final_train_samples}"
            os.makedirs(save_dir, exist_ok=True)
            pipeline_filename = f"pipeline_{self.selected_model_name}.joblib"
            artifacts_filename = f"pipeline_{self.selected_model_name}_artifacts.joblib"

            pipeline_path = os.path.join(save_dir, pipeline_filename)
            artifacts_path = os.path.join(save_dir, artifacts_filename)

            artifacts = {
                "model_name": self.selected_model_name,
                "model_params": self.model_params,
                "n_lags": current_n_lags,
                "trained_on_samples": final_train_samples,
                "features_used": self.features,  # Save the list of features
            }

            joblib.dump(self.pipeline, pipeline_path)
            joblib.dump(artifacts, artifacts_path)
            self.log(f"Model saved to {pipeline_path}")
            self.log(f"Artifacts saved to {artifacts_path}")
            QMessageBox.information(
                self, "Save Successful", f"Model and artifacts saved to:\n{save_dir}"
            )
        except OSError as e:
            self.log(f"Error creating directory or writing file: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Could not create directory or write file: {e}"
            )
        except Exception as e:
            self.log(f"Error saving model: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save model: {e}")

    def on_configure_model(self):
        """Opens a dialog to configure model parameters."""
        self.log("Opening model configuration dialog...")
        selected_model_for_config = self.view.select_model.currentText()

        if not selected_model_for_config or selected_model_for_config == "Select":
            self.log("No model selected to configure.")
            QMessageBox.information(
                self,
                "Configuration",
                "Please select a model first to configure its parameters.",
            )
            return

        # Determine initial parameters for the dialog
        # If params were previously configured for this model (and stored in self.model_params), use them.
        # Otherwise, use defaults for the selected_model_for_config.
        if (
            self.current_model_name_for_params == selected_model_for_config
            and self.model_params
        ):
            initial_params_text = str(self.model_params)
            self.log(
                f"Loading current parameters for {selected_model_for_config} into dialog."
            )
        else:
            default_params = getattr(utils.BestParams(), selected_model_for_config, {})
            initial_params_text = str(default_params)
            self.log(
                f"Loading default parameters for {selected_model_for_config} into dialog."
            )

        dialog = ModelConfigDialog(self, initial_text=initial_params_text)

        if dialog.exec_() == QDialog.Accepted:
            try:
                configured_params = dialog.get_params()
                self.model_params = configured_params
                self.current_model_name_for_params = selected_model_for_config  # Track that these params are for this model
                self.log(
                    f"Parameters updated for {selected_model_for_config}: {self.model_params}"
                )

                if (
                    self.pipeline
                    and self.selected_model_name == selected_model_for_config
                ):
                    self.log(
                        "Note: Re-train the model for these new parameters to take effect on the current pipeline."
                    )
                elif (
                    self.pipeline
                    and self.selected_model_name != selected_model_for_config
                ):
                    self.log(
                        f"Note: Parameters configured for {selected_model_for_config}. Current trained model is {self.selected_model_name}."
                    )

            except ValueError as e:
                QMessageBox.warning(self, "Parameter Error", f"Invalid parameters: {e}")
                self.log(f"Parameter configuration error: {e}")
        else:
            self.log("Model configuration canceled.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("ControlManager with Targets")
    manager = BuildClfManager()
    main_win.setCentralWidget(manager)
    main_win.resize(800, 600)
    main_win.show()
    sys.exit(app.exec_())
