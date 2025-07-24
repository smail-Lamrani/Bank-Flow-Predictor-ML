import sys
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget, QVBoxLayout, QPushButton

from common import run_with_progress


def clean_data(df: pd.DataFrame, callback=None) -> pd.DataFrame:
    # Exemple simple : nettoyage de colonnes numériques en supprimant espaces insécables
    cols = [col for col in df.columns if col != 'date']
    total_steps = len(cols)
    for i, col in enumerate(cols):
        df[col] = df[col].astype(str).str.replace("\u202F", "", regex=False).str.replace("\u00A0", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if callback:
            progress_value = int((i + 1) / total_steps * 100)
            callback(progress_value)
    return df


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Progress Bar with Thread")
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout(self)

        self.button = QPushButton("Start Clean Data", self)
        self.button.clicked.connect(self.start_clean)

        self.layout.addWidget(self.button)

    def start_clean(self):
        # Simule un DataFrame avec colonnes à nettoyer
        n_rows = 10000

# Colonnes avec valeurs contenant espaces insécables
        data = pd.DataFrame({
            'date': pd.date_range(start='2021-01-01', periods=n_rows, freq='D').astype(str),
            'col1': ['123\u202F456'] * n_rows,
            'col2': ['789\u00A0123'] * n_rows,
            'col3': ['1\u202F000'] * n_rows,
            'col4': ['2\u00A0000'] * n_rows,
            'col5': ['345\u202F678'] * n_rows,
            'col6': ['901\u00A0234'] * n_rows,
            'col7': ['567\u202F890'] * n_rows,
            'col8': ['123\u00A0456'] * n_rows,
            'col9': ['789\u202F012'] * n_rows,
            'col10': ['345\u00A0678'] * n_rows,
        })


        self.thread = run_with_progress(self, clean_data, data, feedback=True)
        self.thread.result.connect(self.on_clean_data_result)
        self.thread.error.connect(self.on_error)
        self.thread.finished.connect(self.on_finished)

    def on_clean_data_result(self, df):
        print("Data cleaned:")
        print(df)

    def on_error(self, error_msg):
        print("Error:", error_msg)

    def on_finished(self):
        print("Cleaning finished.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
