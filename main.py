import os
import sys

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QStackedWidget
from PyQt5.uic import loadUi

from common import WorkerThread
from ressources.py_frames.about.about_frame import AboutFrame
from ressources.py_frames.data_preparation.advanced_data_preparation_frame import (
    Advanced_Data_Preparation,
)
from ressources.py_frames.data_preparation.data_exploration_frame import (
    Data_exploration,
)
from ressources.py_frames.data_preparation.visualisation.vis_frame import (
    VisualisationManager,
)
from ressources.py_frames.forcasting.clf.clf_frame import BuildClfManager
from ressources.py_frames.forcasting.forecast.forcast_frame import Forecaster_Manager
from ressources.py_frames.forcasting.reg.reg_frame import BuildRegManager

# from ressources.py_frames.about.about_frame_enhanced_improved import AboutFrameEnhancedImproved as AboutFrame
from ressources.py_frames.home.home_frame import HomeFrame

class MyMainWindow(QMainWindow):
    status_log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Utiliser un chemin absolu pour le fichier UI
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ressources", "main_window.ui"
        )
        loadUi(self.file_path, self)

        # Appliquer le style global à la fenêtre principale
        style_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ressources", "global_style.css"
        )
        try:
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Erreur lors du chargement du style: {e}")

        # Parameters
        self.data = pd.DataFrame()  # used
        self.clients_data = {}  # used
        self.clients_grouped_data = None
        self.clients_info = None
        self.num_obs_client = None
        self.train_data = {}
        self.current_client_id = None  # used

        # Connect the status_log signal to the status slot.
        self.status_log.connect(self.status)

        # Menu and frames --------------------------------------------
        self.stacked_widget = QStackedWidget(self)

        self.home_frame = HomeFrame(self)
        self.data_exploration = Data_exploration(self)
        self.advanced_data_preparation = Advanced_Data_Preparation(
            modify=True, parent=self
        )
        self.build_models = BuildRegManager(self)
        self.Build_classifier_Manager = BuildClfManager(self)
        self.forcast = Forecaster_Manager(self)
        self.visualiser = VisualisationManager(self)
        self.about_frame = AboutFrame(self)

        self.stacked_widget.addWidget(self.home_frame)
        self.stacked_widget.addWidget(self.data_exploration)
        self.stacked_widget.addWidget(self.advanced_data_preparation)
        self.stacked_widget.addWidget(self.build_models)
        self.stacked_widget.addWidget(self.Build_classifier_Manager)
        self.stacked_widget.addWidget(self.forcast)
        self.stacked_widget.addWidget(self.visualiser)
        self.stacked_widget.addWidget(self.about_frame)

        self.setCentralWidget(self.stacked_widget)
        # Main window icon representing the overall app context  # Afficher la page d'accueil par défaut
        self.setWindowTitle(
            "Forecasting Tool Pro - Analyse prédictive avancée"
        )  # Afficher la page d'accueil par défaut

        # Initialize toolbar and set default frame
        self.init_toolbar()
        self.stacked_widget.setCurrentIndex(0)  # Afficher la page d'accueil par défaut


    def get_train_data(self):
        return self.train_data

    def set_train_data(self, train_data):
        self.train_data = train_data

    def get_current_client_id(self):
        return self.current_client_id

    def set_current_client_id(self, client_id):
        self.current_client_id = client_id

    def get_clients_grouped_data(self):
        return self.clients_grouped_data

    def set_clients_grouped_data(self, clients_grouped_data):
        self.clients_grouped_data = clients_grouped_data

    def get_num_obs_client(self):
        return self.num_obs_client

    def set_num_obs_client(self, num_obs_client):
        self.num_obs_client = num_obs_client

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_clients_data(self, client_id):
        return self.clients_data.get(client_id, None)

    def set_clients_data(self, client_id, client_data):
        self.clients_data[client_id] = client_data

    def get_clients_info(self):
        return self.clients_info

    def set_clients_info(self, clients_info):
        self.clients_info = clients_info

    def init_toolbar(self):
        menu_bar = self.menuBar

        # Menu 1: Home
        self.home_menu = menu_bar.addMenu("Home")
        home_action = QAction("Accueil", self)
        self.home_menu.addAction(home_action)
        home_action.triggered.connect(lambda x: self.set_current_frame(HomeFrame))

        # Menu 2: Data
        self.main_menu_data = menu_bar.addMenu("Data")

        # Action 1: Load Data
        explore_data_action = QAction("Load Data", self)
        self.main_menu_data.addAction(explore_data_action)
        explore_data_action.triggered.connect(
            lambda x: self.set_current_frame(Data_exploration)
        )

        # Action 2: Advanced Data Preparation
        advanced_data_action = QAction("Advanced Data Preparation", self)
        self.main_menu_data.addAction(advanced_data_action)
        advanced_data_action.triggered.connect(
            lambda x: self.set_current_frame(Advanced_Data_Preparation)
        )
        # Action 3: Advanced Data Preparation
        vis_action = QAction("Data Visualisation", self)
        self.main_menu_data.addAction(vis_action)
        vis_action.triggered.connect(
            lambda x: self.set_current_frame(VisualisationManager)
        )

        # Menu 3: Models
        self.main_menu_models = menu_bar.addMenu("Models")

        # Action: Build Regressor
        build_models_action = QAction("Build Regressor", self)
        self.main_menu_models.addAction(build_models_action)
        build_models_action.triggered.connect(
            lambda x: self.set_current_frame(BuildRegManager)
        )
        # Action: Build Classifier
        build_classifier_action = QAction("Build Classifier", self)
        self.main_menu_models.addAction(build_classifier_action)
        build_classifier_action.triggered.connect(
            lambda x: self.set_current_frame(BuildClfManager)
        )

        # Menu 4: Forecast
        self.main_menu_forecast = menu_bar.addMenu("Forecast")
        # Action: Forecast
        forecast_action = QAction("Forecast", self)
        self.main_menu_forecast.addAction(forecast_action)
        forecast_action.triggered.connect(
            lambda x: self.set_current_frame(Forecaster_Manager)
        )

        # Menu 5: About
        self.about_menu = menu_bar.addMenu("About")
        about_action = QAction("About", self)
        self.about_menu.addAction(about_action)
        about_action.triggered.connect(
            lambda x: self.set_current_frame(AboutFrame)
        )

        # Menu 6: Help
        # self.help_menu = menu_bar.addMenu("Help")
        # help_action = QAction("Help", self)
        # self.help_menu.addAction(help_action)

    def set_current_frame(self, cls):
        self.update_instance_data()
        for index in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(index)
            if isinstance(widget, cls):
                self.stacked_widget.setCurrentWidget(widget)
                return True
        return False

    def get_data(self):
        """Récupère les données actuelles."""
        try:
            # Essayer d'abord de récupérer les données depuis data_exploration
            if (
                hasattr(self, "data_exploration")
                and hasattr(self.data_exploration, "data")
                and self.data_exploration.data is not None
            ):
                return self.data_exploration.data

            # Sinon, essayer de récupérer les données depuis advanced_data_preparation
            if hasattr(self, "advanced_data_preparation") and hasattr(
                self.advanced_data_preparation, "data_manager"
            ):
                if (
                    hasattr(self.advanced_data_preparation.data_manager, "data")
                    and self.advanced_data_preparation.data_manager.data is not None
                ):
                    return self.advanced_data_preparation.data_manager.data

            # Si aucune donnée n'est disponible, retourner un DataFrame vide
            import pandas as pd

            return pd.DataFrame()
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            import pandas as pd

            return pd.DataFrame()

    def set_data(self, data):
        """Définit les données actuelles."""
        try:
            # Stocker les données dans data_exploration
            if not hasattr(self, "data"):
                self.data = data
            else:
                self.data = data

            # Mettre à jour les données dans data_exploration
            if hasattr(self, "data_exploration"):
                self.data_exploration.data = data

            # Mettre à jour les données dans advanced_data_preparation
            if hasattr(self, "advanced_data_preparation") and hasattr(
                self.advanced_data_preparation, "data_manager"
            ):
                self.advanced_data_preparation.data_manager.data = data
                self.advanced_data_preparation.data_manager.original_data = data

            # Mettre à jour les variables disponibles pour la visualisation
            if hasattr(self, "about_frame") and hasattr(
                self.about_frame, "update_variables"
            ):
                self.about_frame.update_variables()

            print(
                f"Données définies avec succès: {data.shape[0]} lignes, {data.shape[1]} colonnes"
            )
        except Exception as e:
            print(f"Erreur lors de la définition des données: {e}")

    def update_instance_data(self):
        if self.get_data().empty:
            return
        self.advanced_data_preparation.data_manager.data = self.get_data()
        self.advanced_data_preparation.data_manager.original_data = self.get_data()
        self.advanced_data_preparation.update_row_selector(5)
        self.advanced_data_preparation.update_table()
        self.data_exploration.update_table()

        # Mettre à jour les variables disponibles pour la visualisation
        if hasattr(self, "about_frame") and hasattr(
            self.about_frame, "update_variables"
        ):
            self.about_frame.update_variables()

    def status(self, text):
        # Afficher le message de statut
        self.statusBar().showMessage(f"  {text}")


if __name__ == "__main__":
    app = QApplication([])
    main_window = MyMainWindow()
    main_window.show()
    app.exec_()
