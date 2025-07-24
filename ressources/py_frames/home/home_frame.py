import os
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSignal, QObject
from ressources.py_frames.data_preparation.data_exploration_frame import Data_exploration
from ressources.py_frames.home.home_view import HomeView
from ressources.py_frames.forcasting.clf.clf_frame import BuildClfManager
from ressources.py_frames.forcasting.forecast.forcast_frame import Forecaster_Manager
from ressources.py_frames.forcasting.reg.reg_frame import BuildRegManager
from PyQt5.QtWebChannel import QWebChannel


from PyQt5.QtCore import QObject, pyqtSlot

class JSBridge(QObject):
    def __init__(self, parent):
        super(JSBridge, self).__init__()
        self.parent = parent

    @pyqtSlot()
    def navigateToDataExploration(self):
        self.parent.navigateToDataExploration()

    @pyqtSlot()
    def navigateToForecast(self):
        self.parent.navigateToForecast()

    @pyqtSlot()
    def navigateToClf(self):
        self.parent.navigateToClf()

    @pyqtSlot()
    def navigateToReg(self):
        self.parent.navigateToReg()


class HomeFrame(QFrame):
    update_ext_log = pyqtSignal(str)
    
    def __init__(self, parent):
        super(HomeFrame, self).__init__(parent)
        self.parent = parent
        
        # Créer le layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Créer une instance de HomeView
        self.home_view = HomeView(self)
        self.home_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.home_view)
        
        self.setLayout(layout)
        # Créer une instance de JSBridge
        self.bridge = JSBridge(self)
        # Créer le canal et enregistrer l'objet
        self.channel = QWebChannel()
        self.channel.registerObject('bridge', self.bridge)
        # Lier le canal au web view
        self.home_view.web_view.page().setWebChannel(self.channel)
        
        # Appliquer le style global
        style_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "global_style.css")
        try:
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Erreur lors du chargement du style: {e}")
            
        # Configurer les méthodes de navigation
        self.setupNavigation()
        
    def setupNavigation(self):
        """Configure les méthodes de navigation pour les liens rapides"""
        # Pour QWebEngineView, nous utilisons des méthodes natives de navigation
        # Les fonctions JavaScript appelleront directement ces méthodes
        
    def navigateToDataExploration(self):
        """Navigue vers la page d'exploration des données"""
        if hasattr(self.parent, 'set_current_frame'):
            self.parent.set_current_frame(Data_exploration)
            
            
    def navigateToForecast(self):
        """Navigue vers la page de prévisions"""
        if hasattr(self.parent, 'set_current_frame'):
            self.parent.set_current_frame(Forecaster_Manager)
    
    def navigateToClf(self):
        """Navigue vers la page de prévisions"""
        if hasattr(self.parent, 'set_current_frame'):
            self.parent.set_current_frame(BuildClfManager)

    def navigateToReg(self):
        """Navigue vers la page de prévisions"""
        if hasattr(self.parent, 'set_current_frame'):
            self.parent.set_current_frame(BuildRegManager)