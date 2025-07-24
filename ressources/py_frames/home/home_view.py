import os

from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QFrame, QSizePolicy, QVBoxLayout


class HomeView(QFrame):
    def __init__(self, parent=None):
        super(HomeView, self).__init__(parent)
        self.setObjectName("HomeView")

        # Charger le style CSS global (pour la QFrame)
        css_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "home_style.css"
        )
        self.load_styles(css_path)

        # Initialisation de l'interface
        self.setupUi()

    def load_styles(self, style_path):
        """Charge le fichier de style CSS externe pour le QFrame."""
        try:
            with open(style_path, "r") as f:
                style = f.read()
                self.setStyleSheet(style)
        except Exception as e:
            print(f"Erreur lors du chargement du style: {e}")

    def setupUi(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Utilisation de QWebEngineView
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Contenu HTML
        html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Accueil – Bank of Africa</title>
    <style>
        :root {
            --primary-color: #002D62;
            --secondary-color: #0056b3;
            --accent-color: #FFD700;
            --text-color: #333;
            --light-text: #666;
            --bg-color: #f8f9fa;
            --card-bg: #fff;
            --border-radius: 8px;
            --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text-color);
            background-color: var(--bg-color);
            line-height: 1.6;
            position: relative;
            overflow-x: hidden;
        }
        
        .background-logo {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            max-width: 800px;
            opacity: 0.04;
            z-index: -1;
            pointer-events: none;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: var(--primary-color);
            font-size: 2.2rem;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .header p {
            color: var(--light-text);
            font-size: 1.1rem;
            max-width: 700px;
            margin: 0 auto;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            padding: 30px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .card h2 {
            color: var(--primary-color);
            font-size: 1.5rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        
        .card h2 span {
            margin-right: 12px;
        }
        
        .feature-list {
            list-style: none;
            margin-top: 20px;
        }
        
        .feature-list li {
            margin-bottom: 12px;
            padding-left: 28px;
            position: relative;
        }
        
        .feature-list li:before {
            content: "";
            position: absolute;
            left: 0;
            top: 8px;
            width: 8px;
            height: 8px;
            background-color: var(--accent-color);
            border-radius: 50%;
        }
        
        .about-section {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 40px;
            border-radius: var(--border-radius);
            margin-bottom: 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .about-content {
            flex: 1;
        }
        
        .about-section h2 {
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: white;
        }
        
        .about-section p {
            margin-bottom: 15px;
            line-height: 1.7;
        }
        
        .about-logo {
            flex: 0 0 150px;
            margin-left: 30px;
        }
        
        .about-logo img {
            max-width: 100%;
            border-radius: 50%;
            background: white;
            padding: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .action-button {
            background-color: var(--card-bg);
            border: 1px solid #eaeaea;
            border-radius: var(--border-radius);
            padding: 25px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: var(--box-shadow);
        }
        
        .action-button:hover {
            border-color: var(--accent-color);
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        
        .action-button span {
            font-size: 2rem;
            margin-bottom: 15px;
            color: var(--primary-color);
        }
        
        .action-button h3 {
            font-size: 1.2rem;
            margin-bottom: 10px;
            color: var(--primary-color);
        }
        
        .action-button p {
            font-size: 0.9rem;
            color: var(--light-text);
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--light-text);
            font-size: 0.9rem;
            border-top: 1px solid #eaeaea;
            margin-top: 40px;
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .quick-actions {
                grid-template-columns: 1fr;
            }
            
            .about-section {
                flex-direction: column;
                text-align: center;
            }
            
            .about-logo {
                margin: 20px 0 0 0;
            }
        }
    </style>
</head>
<body>
    <!-- Logo en arrière-plan -->
    <img class="background-logo" src="https://www.ir-bankofafrica.ma/sites/default/files/styles/actualite/public/default_images/imgpsh_fullsize_anim.png?itok=UUjnntlS" alt="Bank of Africa Logo Background">
    
    <div class="container">
        <div class="header">
            <h1>Application de Prévision des Flux Digitaux</h1>
            <p>Une solution avancée pour analyser et prédire les comportements transactionnels des clients de Bank of Africa</p>
        </div>
        
        <div class="content-grid">
            <div class="card">
                <h2><span>📊</span> Analyse Prédictive</h2>
                <p>Utilisez des algorithmes avancés pour prédire avec précision les comportements transactionnels futurs de vos clients.</p>
                <ul class="feature-list">
                    <li>Prévision du nombre de transactions par canal</li>
                    <li>Estimation des montants transactionnels</li>
                    <li>Détection des tendances émergentes</li>
                </ul>
            </div>
            
            <div class="card">
                <h2><span>👥</span> Segmentation Client</h2>
                <p>Identifiez des segments de clientèle basés sur leurs comportements digitaux pour des stratégies marketing ciblées.</p>
                <ul class="feature-list">
                    <li>Regroupement par comportement d'utilisation</li>
                    <li>Identification des clients à forte valeur</li>
                    <li>Analyse des préférences de canal</li>
                </ul>
            </div>
        </div>
        
        <div class="about-section">
            <div class="about-content">
                <h2>Bank of Africa</h2>
                <p>Présente dans plus de 30 pays, Bank of Africa est l'un des groupes bancaires les plus dynamiques d'Afrique, offrant des services financiers innovants et accessibles.</p>
                <p>Notre mission est de soutenir le développement économique à travers le continent tout en fournissant des solutions bancaires de classe mondiale à nos clients.</p>
            </div>
            <div class="about-logo">
                <img src="https://www.ir-bankofafrica.ma/sites/default/files/styles/actualite/public/default_images/imgpsh_fullsize_anim.png?itok=UUjnntlS" alt="Bank of Africa Logo">
            </div>
        </div>
        
        <div class="quick-actions">
            <div class="action-button" onclick="window.parent.navigateToDataExploration()">
                <span>🔍</span>
                <h3>Explorer les Données</h3>
                <p>Visualisez et analysez les données transactionnelles</p>
            </div>
            
            <div class="action-button" onclick="window.parent.navigateToClf()">
                <span>🧠</span>
                <h3>Construire des Classifiers</h3>
                <p>Créez des modèles de detection de l'activité </p>
            </div>

            <div class="action-button" onclick="window.parent.navigateToReg()">
                <span>🧠</span>
                <h3>Construire des Regresseurs</h3>
                <p>Créez des modèles prédictifs</p>
            </div>
            
            <div class="action-button" onclick="window.parent.navigateToForecast()">
                <span>📈</span>
                <h3>Générer des Prévisions</h3>
                <p>Obtenez des prédictions précises sur les flux futurs</p>
            </div>
        </div>
        
        <div class="footer">
            © 2024 Bank of Africa | Application de Prévision des Flux Digitaux | Tous droits réservés
        </div>
    </div>
    
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
            document.addEventListener("DOMContentLoaded", function () {
                new QWebChannel(qt.webChannelTransport, function (channel) {
                    window.bridge = channel.objects.bridge;

                    // Redéfinir les fonctions de navigation
                    window.parent.navigateToDataExploration = function() {
                        bridge.navigateToDataExploration();
                    };
                    window.parent.navigateToForecast = function() {
                        bridge.navigateToForecast();
                    };
                    window.parent.navigateToClf = function() {
                        bridge.navigateToClf();
                    };
                    window.parent.navigateToReg = function() {
                        bridge.navigateToReg();
                    };

                });
            });
        </script>

</body>
</html>"""

        # Charger le contenu HTML dans le QWebEngineView
        self.web_view.setHtml(html_content, QUrl("file://"))

        # Ajouter au layout
        layout.addWidget(self.web_view)
