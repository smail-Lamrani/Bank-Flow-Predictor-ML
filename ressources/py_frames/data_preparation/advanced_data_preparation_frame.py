import ast
import os
import pandas as pd
from PyQt5.QtWidgets import QCompleter
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
    QFileDialog,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class ScatterPlotDialog(QDialog):
    def __init__(self, data: pd.DataFrame, col_x: str, col_y: str, parent=None):
        super().__init__(parent)
        self.data = data
        self.col_x = col_x
        self.col_y = col_y
        self.setWindowTitle(f"Scatter Plot: {col_x} vs {col_y}")
        self.resize(600, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.plot_scatter()
        self.setLayout(layout)

    def plot_scatter(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = self.data[self.col_x]
        y = self.data[self.col_y]
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel(self.col_x)
        ax.set_ylabel(self.col_y)
        ax.set_title(f"{self.col_x} vs {self.col_y}")
        self.canvas.draw()


class SafeVisitor(ast.NodeVisitor):
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.safe = True

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.allowed:
            self.safe = False
        self.generic_visit(node)

    def visit_Import(self, node):
        self.safe = False

    def visit_ImportFrom(self, node):
        self.safe = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "df":
                pass
            elif isinstance(node.func.value, ast.Subscript):
                if not (
                    isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "df"
                ):
                    self.safe = False
            else:
                self.safe = False
        else:
            self.safe = False
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id != "df":
                    self.safe = False
            elif isinstance(target, ast.Subscript):
                if not (isinstance(target.value, ast.Name) and target.value.id == "df"):
                    self.safe = False
            else:
                self.safe = False
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        orig_allowed = self.allowed.copy()
        for arg in node.args.args:
            self.allowed.add(arg.arg)
        self.generic_visit(node)
        self.allowed = orig_allowed

    def visit_Lambda(self, node):
        orig_allowed = self.allowed.copy()
        for arg in node.args.args:
            self.allowed.add(arg.arg)
        self.generic_visit(node)
        self.allowed = orig_allowed


def is_safe_code(code: str) -> bool:
    try:
        tree = ast.parse(code, mode="exec")
        visitor = SafeVisitor(allowed={"df"})
        visitor.visit(tree)
        return visitor.safe
    except Exception:
        return False


class DataManager:
    """
    Encapsulates the DataFrame and all operations on it.
    This class handles transformations, filtering, renaming, dropping,
    exporting, reverting changes and computing correlations.
    """

    def __init__(self, data: pd.DataFrame, parent=None):
        self.original_data = data.copy()
        self.data = data.copy()
        self.parent = parent

    def update_data(self):
        self.parent.set_data(self.data)

    def apply_transformation(self, code: str, safe_exec: bool = True) -> None:
        if safe_exec and not is_safe_code(code):
            raise ValueError("Invalid code! Only transformations on 'df' are allowed.")
        local_vars = {"df": self.data}
        exec(code, {"__builtins__": {}}, local_vars)
        self.data = local_vars["df"]

    def rename_column(self, old_name: str, new_name: str) -> None:
        if not new_name:
            raise ValueError("New column name cannot be empty.")
        if new_name in self.data.columns:
            raise ValueError("Column name already exists.")
        self.data.rename(columns={old_name: new_name}, inplace=True)

    def drop_column(self, column: str) -> None:
        if column not in self.data.columns:
            raise ValueError("Column not found.")
        self.data.drop(columns=[column], inplace=True)

    def batch_rename(self, new_names: list) -> None:
        if len(new_names) != len(self.data.columns):
            raise ValueError(
                "The number of new column names must match the number of existing columns."
            )
        if len(new_names) != len(set(new_names)):
            raise ValueError("Duplicate column names are not allowed.")
        rename_dict = dict(zip(list(self.data.columns), new_names))
        self.data.rename(columns=rename_dict, inplace=True)

    def batch_drop(self, columns: list) -> None:
        self.data.drop(columns=columns, inplace=True)

    def apply_filter(self, filter_text: str) -> None:
        self.data = self.data.query(filter_text)

    def revert_changes(self) -> None:
        self.data = self.original_data.copy()

    def export_data(self, file_path: str) -> None:
        if file_path.endswith(".csv"):
            self.data.to_csv(file_path, index=False)
        elif file_path.endswith(".xlsx"):
            self.data.to_excel(file_path, index=False)
        elif file_path.endswith(".pkl"):
            self.data.to_pickle(file_path)
        else:
            raise ValueError("Unsupported file format.")

class BatchRenameDialog(QDialog):
    def __init__(self, current_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Rename Columns")
        self.current_columns = current_columns
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )
        label = QLabel("Edit the column names (one per line):")
        layout.addWidget(label)
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText("\n".join(self.current_columns))
        layout.addWidget(self.text_edit)
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_new_columns(self):
        return [
            name.strip()
            for name in self.text_edit.toPlainText().splitlines()
            if name.strip()
        ]


class BatchDropDialog(QDialog):
    def __init__(self, current_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Drop Columns")
        self.current_columns = current_columns
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )
        label = QLabel("Select columns to drop (check them):")
        layout.addWidget(label)
        self.list_widget = QListWidget()
        for col in self.current_columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_columns_to_drop(self):
        cols_to_drop = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                cols_to_drop.append(item.text())
        return cols_to_drop


class TransformationHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transformation Help")
        self.setMinimumSize(600, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )
        help_text = """
            📌 **Available Functions to Use**
        -----------------------------------------
        🟢 **Basic Statistics**
        - `df['mean_A'] = df['A'].mean()`  → Column mean
        - `df['median_A'] = df['A'].median()` → Column median
        - `df['std_A'] = df['A'].std()` → Standard deviation
        - `df['var_A'] = df['A'].var()` → Variance

        🟢 **Normalization & Scaling**
        - `df['A_zscore'] = (df['A'] - df['A'].mean()) / df['A'].std()` → Z-score
        - `df['A_scaled'] = (df['A'] - df['A'].min()) / (df['A'].max() - df['A'].min())` → Min-Max Scaling

        🟢 **Rolling Statistics**
        - `df['A_rolling_mean'] = df['A'].rolling(window=3).mean()` → Rolling Mean
        - `df['A_rolling_std'] = df['A'].rolling(window=3).std()` → Rolling Standard Deviation

        🟢 **Exponential Weighted Moving Average**
        - `df['A_ewm'] = df['A'].ewm(span=3, adjust=False).mean()` → EWMA (Smooths Trends)

        🟢 **Transformations**
        - `df['A_log'] = np.log(df['A'] + 1)` → Log Transform
        - `df['A_squared'] = df['A'] ** 2` → Squared Feature
        - `df['A_cubed'] = df['A'] ** 3` → Cubed Feature

        🟢 **Feature Interactions**
        - `df['A_B_ratio'] = df['A'] / (df['B'] + 1e-6)` → Ratio (Avoids Zero-Division)
        - `df['A_B_sum'] = df['A'] + df['B']` → Sum of Two Columns
        - `df['A_B_mult'] = df['A'] * df['B']` → Multiplication of Columns
        -----------------------------------------
        ℹ️ You can enter these functions in the text field and apply them to the DataFrame.
        """
        text_edit = QTextEdit()
        text_edit.setPlainText(help_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
        self.setLayout(layout)


class Advanced_Data_Preparation(QFrame):
    def __init__(self, modify=True, parent=None):
        super().__init__(parent)
        self.data_manager = DataManager(parent.get_data(), parent)
        self.modify = modify
        self.updating_table = False  # Prevent signals during programmatic updates
        self.safe_exec = True
        self.filter_restoring_data = None
        self.init_ui()

        # After initializing self.filter_input:
        filter_completer = QCompleter(list(self.data_manager.data.columns))
        filter_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.filter_input.setCompleter(filter_completer)
        self.filter_input.installEventFilter(self)
        self.pandas_functions = [
            "df",
            "median",
            "var",
            "astype",
            "fillna",
            "dropna",
            "groupby",
            "sort_values",
            "apply",
            "replace",
            "isin",
            "between",
            "query",
            "clip",
            "abs",
            "round",
            "sum",
            "mean",
            "max",
            "min",
            "std",
            "value_counts",
            "unique",
            "nunique",
            "rename",
            "drop",
            "loc",
            "iloc",
            "head",
            "tail",
        ]

        all_completions = list(self.data_manager.data.columns) + self.pandas_functions
        trans_completer = QCompleter(all_completions)
        trans_completer.setCaseSensitivity(Qt.CaseInsensitive)
        trans_completer.setFilterMode(Qt.MatchContains)
        trans_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.transform_input.setCompleter(trans_completer)
        self.transform_input.installEventFilter(self)

    def init_ui(self):
        self.setWindowTitle("Advanced data preparation")
        # self.setGeometry(300, 200, 1000, 500)
        self.setMinimumSize(800, 400)
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )
        main_layout = QVBoxLayout()

        # Filtering Panel & Column selection for analysis.
        v_layout = QVBoxLayout()
        filter_layout = QHBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter filter (e.g., client == 50000000)")
        self.btn_filter = QPushButton("Filter")
        self.btn_filter.clicked.connect(self.apply_filter)
        filter_layout.addWidget(
            QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken)
        )
        self.column_select = QComboBox()
        self.column_select.setFixedWidth(200)
        self.column_select.addItem("Get Summary")
        self.column_select.addItems(list(self.data_manager.data.columns))
        self.column_select.currentIndexChanged.connect(self.update_column_summary)
        filter_layout.addWidget(self.column_select)
        filter_layout.addWidget(self.filter_input)
        filter_layout.addWidget(self.btn_filter)

        # Modification Panel (if modify=True).
        if self.modify:
            # Transformation Panel.
            transform_layout = QHBoxLayout()
            self.rename_drop_combo = QComboBox()
            self.rename_drop_combo.setFixedWidth(200)
            self.rename_drop_combo.addItem("Process columns")
            self.rename_drop_combo.addItem("Rename columns")
            self.rename_drop_combo.addItem("Drop columns")
            self.rename_drop_combo.currentIndexChanged.connect(self.ren_drop_columns)

            transform_layout.addWidget(self.rename_drop_combo)

            self.transform_input = QLineEdit()
            self.transform_input.setPlaceholderText(
                "Custom python code (e.g., df['new_col'] = df['col1']*10)"
            )
            transform_layout.addWidget(self.transform_input)
            self.btn_apply_transform = QPushButton("Apply")
            self.btn_apply_transform.clicked.connect(self.apply_transformation)
            transform_layout.addWidget(self.btn_apply_transform)

            v_layout.addLayout(transform_layout)
            v_layout.addLayout(filter_layout)

            main_layout.addLayout(v_layout)

        # Table.
        self.table_layout = QVBoxLayout()
        self.table = QTableWidget()
        if self.modify:
            self.table.setEditTriggers(QTableWidget.AllEditTriggers)
            self.table.cellChanged.connect(self.cell_edited)

        self.table_layout.addWidget(self.table)
        main_layout.addLayout(filter_layout)
        main_layout.addLayout(self.table_layout)

        # Bottom Buttons.
        button_layout = QHBoxLayout()
        # button_layout.addStretch()
        row_layout = QHBoxLayout()
        self.row_label = QLabel("Rows to Display:")
        self.row_selector = QSpinBox()
        self.row_selector.setMinimum(1)
        self.row_selector.setMaximum(len(self.data_manager.data))
        self.row_selector.setValue(min(5, len(self.data_manager.data)))
        self.row_selector.valueChanged.connect(self.update_table)
        row_layout.addWidget(self.row_label)
        row_layout.addWidget(self.row_selector)
        self.rows_num = QLabel(" Rows max : " + str(len(self.data_manager.data)))
        row_layout.addWidget(self.rows_num)
        row_layout.addStretch()
        button_layout.addLayout(row_layout)
        self.btn_update_data = QPushButton("Update Data")
        self.btn_update_data.clicked.connect(self.update_data)
        button_layout.addWidget(self.btn_update_data)
        if self.modify:
            self.btn_save_changes = QPushButton("Export Changes")
            self.btn_revert_changes = QPushButton("Revert Changes")
            self.btn_save_changes.clicked.connect(self.save_changes)
            self.btn_revert_changes.clicked.connect(self.revert_changes)
            button_layout.addWidget(self.btn_save_changes)
            button_layout.addWidget(self.btn_revert_changes)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.update_table()

    def ren_drop_columns(self):
        selected_option = self.rename_drop_combo.currentText()
        if selected_option == "Rename columns":
            self.open_batch_rename_dialog()
        elif selected_option == "Drop columns":
            self.open_batch_drop_dialog()
        self.rename_drop_combo.setCurrentIndex(0)

    def update_row_selector(self, init_value = None):
        self.row_selector.setMaximum(len(self.data_manager.data))
        if init_value:
            self.row_selector.setValue(init_value)
        self.rows_num.setText(" Rows max : " + str(len(self.data_manager.data)))

    def update_table(self):
        if self.column_select.currentText() != "Get Summary":
            self.update_column_summary()
            return
        if self.modify:
            # Update drop and rename dialog lists if needed.
            pass
        self.column_select.currentIndexChanged.disconnect(self.update_column_summary)

        self.column_select.clear()
        self.column_select.addItem("Get Summary")
        self.column_select.addItems(list(self.data_manager.data.columns))
        self.updating_table = True
        num_rows = min(self.row_selector.value(), len(self.data_manager.data))
        num_cols = len(self.data_manager.data.columns)
        
        self.table.clear()
        self.table.setRowCount(num_rows)
        self.table.setColumnCount(num_cols)
        self.table.setHorizontalHeaderLabels(list(self.data_manager.data.columns))
        for row in range(num_rows):
            for col in range(num_cols):
                item = QTableWidgetItem(str(self.data_manager.data.iloc[row, col]))
                if self.modify:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.updating_table = False
        self.column_select.currentIndexChanged.connect(self.update_column_summary)

    def update_column_summary(self):
        selected_column = self.column_select.currentText()
        if selected_column and selected_column not in (
            "Select Column to Analyze",
            "Get Summary",
        ):
            column_data = (
                self.data_manager.data[selected_column].describe().reset_index()
            )
            self.updating_table = True
            self.table.clear()
            self.table.setRowCount(len(column_data))
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Statistic", "Value"])
            for row in range(len(column_data)):
                self.table.setItem(
                    row, 0, QTableWidgetItem(str(column_data.iloc[row, 0]))
                )
                self.table.setItem(
                    row, 1, QTableWidgetItem(str(column_data.iloc[row, 1]))
                )
            self.updating_table = False
        else:
            self.update_table()

    def apply_filter(self):
        filter_text = self.filter_input.text().strip()
        if not filter_text:
            QMessageBox.warning(self, "Warning", "Please enter a filter condition.")
            return
        try:
            self.data_manager.apply_filter(filter_text)
            self.update_row_selector()
            self.update_table()
            QMessageBox.information(
                self, "Filter Applied", "Filter applied successfully."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Invalid filter expression: {str(e)}"
            )
        
    def cell_edited(self, row, column):
        if self.updating_table:
            return
        try:
            new_value = self.table.item(row, column).text()
            col_name = list(self.data_manager.data.columns)[column]
            original_dtype = self.data_manager.original_data[col_name].dtype
            if pd.api.types.is_numeric_dtype(original_dtype):
                try:
                    new_value = float(new_value)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Invalid numeric value at row {row+1}, column '{col_name}'.",
                    )
                    return
            if row < len(self.data_manager.data):
                self.data_manager.data.iat[row, column] = new_value
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update cell: {str(e)}")

    def open_batch_rename_dialog(self):
        dialog = BatchRenameDialog(list(self.data_manager.data.columns), self)
        if dialog.exec_() == QDialog.Accepted:
            new_names = dialog.get_new_columns()
            try:
                reply = QMessageBox.question(
                    self,
                    "Confirm Batch Rename",
                    "Are you sure you want to apply batch renaming?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                self.data_manager.batch_rename(new_names)
                self.update_table()
                QMessageBox.information(
                    self, "Success", "Columns renamed successfully in batch."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to apply batch rename: {str(e)}"
                )

    def open_batch_drop_dialog(self):
        dialog = BatchDropDialog(list(self.data_manager.data.columns), self)
        if dialog.exec_() == QDialog.Accepted:
            cols_to_drop = dialog.get_columns_to_drop()
            if not cols_to_drop:
                QMessageBox.information(
                    self, "No Selection", "No columns were selected for dropping."
                )
                return
            reply = QMessageBox.question(
                self,
                "Confirm Batch Drop",
                f"Are you sure you want to drop the following columns?\n{', '.join(cols_to_drop)}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                self.data_manager.batch_drop(cols_to_drop)
                self.update_table()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Columns dropped successfully: {', '.join(cols_to_drop)}.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to drop columns: {str(e)}")

    def open_transformation_help_dialog(self):
        dialog = TransformationHelpDialog(self)
        dialog.exec_()

    def apply_transformation(self):
        user_code = self.transform_input.text().strip()
        try:
            self.data_manager.apply_transformation(user_code, safe_exec=self.safe_exec)
            self.update_row_selector()
            self.update_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_changes(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Modified Data",
            "",
            "Pickle Files (*.pkl);;Excel Files (*.xlsx);;CSV Files (*.csv)",
            options=options,
        )
        if file_path:
            try:
                self.data_manager.export_data(file_path)
                QMessageBox.information(
                    self, "Success", f"Modified data saved successfully: {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save modified data: {str(e)}"
                )

    def revert_changes(self):
        reply = QMessageBox.question(
            self,
            "Confirm Revert",
            "Are you sure you want to revert changes to the original data?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.data_manager.revert_changes()
        self.update_row_selector()
        self.update_table()
        QMessageBox.information(
            self, "Reverted", "Changes have been reverted to the original data."
        )

    def update_data(self):
        reply = QMessageBox.question(
            self,
            "Confirm Update",
            "Are you sure you want to update changes to the original loaded data?\n(The main data will be updated untill you re-load an other data)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        QMessageBox.information(self, "Updated", "Data has been updated.")
        self.data_manager.update_data()
