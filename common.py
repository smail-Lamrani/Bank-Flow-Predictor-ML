from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout
import traceback # Import the traceback module


class ProgressBarDialog(QDialog):
    def __init__(self, parent=None, message="Veuillez patienter...", cancellable=False):
        super().__init__(parent)
        self.setWindowTitle("Traitement en cours")
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint)
        # 1) Modalité application
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(350, 140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # 3) Message personnalisable
        self.message_label = QLabel(message, self)
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(50)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        # 2) Bouton Annuler optionnel
        if cancellable:
            self.cancel_button = QPushButton("Annuler", self)
            layout.addWidget(self.cancel_button)
        else:
            self.cancel_button = None


class WorkerThread(QThread):
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, func, *args, feedback=False, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.feedback = feedback

    @pyqtSlot()
    def run(self):
        self.started.emit()

        def progress_callback(i):
            self.progress.emit(i)

        if self.feedback:
            self.kwargs["callback"] = progress_callback

        try:
            result = self.func(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            # Create a more detailed error message
            error_type = type(e).__name__
            error_message = str(e)
            tb_str = traceback.format_exc() # Get the full traceback as a string
            detailed_error_message = f"Error Type: {error_type}\nMessage: {error_message}\n\nTraceback:\n{tb_str}"
            self.error.emit(detailed_error_message)
        finally:
            self.finished.emit()


def run_with_progress(
    parent, func, *args, feedback=False, message=None, cancellable=False, **kwargs
):
    # Boîte de dialogue
    dialog = ProgressBarDialog(
        parent=parent,
        message=message or "Veuillez patienter pendant le traitement...",
        cancellable=cancellable,
    )

    # Thread de travail
    thread = WorkerThread(func, *args, feedback=feedback, **kwargs)
    # Connexions
    thread.progress.connect(dialog.progress_bar.setValue)
    thread.started.connect(dialog.show)
    thread.finished.connect(dialog.close)
    thread.error.connect(dialog.close)

    # Si on peut annuler, on relie le bouton Annuler
    if cancellable and dialog.cancel_button:
        dialog.cancel_button.clicked.connect(thread.terminate)

    thread.start()
    return thread

