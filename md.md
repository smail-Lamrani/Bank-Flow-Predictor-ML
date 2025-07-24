| Critère                 | Modèle leader     | Commentaire                                                   |
| ----------------------- | ----------------- | ------------------------------------------------------------- |
| **MAE**                 | **KNN** (0 .285)  | Meilleure erreur absolue sur les trois indices.               |
| **QLIKE**               | **HAR** (0 .253)  | Plus faible score de variance, devant même les RF et KNN.     |
| **Compromis global**    | **KNN**           | 1ᵉ en MAE, 3ᵉ en QLIKE ⇒ meilleur équilibre.                  |
| **Alternative robuste** | **Random Forest** | Premier en QLIKE parmi les purs modèles ML, troisième en MAE. |
