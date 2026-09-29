# TP Prediction automobile

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Objectif du projet

Ce projet a pour objectif de prédire le prix d'un véhicule automobile à partir de ses caractéristiques.

Il s'agit d'un problème de régression supervisée réalisé dans le cadre du cours Processus et écosystème. Le projet applique une démarche MLOps simple allant de la préparation des données jusqu'à l'utilisation du modèle dans une application.

La démarche comprend :

- la préparation et l'exploration des données ;
- l'entraînement de plusieurs modèles et configurations ;
- le suivi des expérimentations avec MLflow ;
- la comparaison des performances à l'aide des métriques MAE, RMSE et R² ;
- la sélection et la sauvegarde du meilleur modèle ;
- l'exposition du modèle avec une API FastAPI ;
- la création d'une interface Streamlit communiquant avec l'API par requête HTTP.

## Démarche du projet

Le projet suit le processus général suivant :

**Données → Préparation → Entraînement → Expérimentations MLflow → Comparaison → Sélection du meilleur modèle → FastAPI → Streamlit**



## Project Organization

```
├── api
│   ├── __init__.py
│   └── main.py                     <- API FastAPI et route POST /predict
│
├── app
│   └── streamlit_app.py            <- Interface Streamlit communiquant avec FastAPI
│
├── data
│   ├── external                    <- Données provenant de sources externes
│   ├── interim
│   │   └── vehicles_clean.csv      <- Dataset nettoyé utilisé pour la modélisation
│   ├── processed                   <- Données finales éventuellement produites
│   └── raw                         <- Données originales
│
├── models
│   └── best_model.joblib           <- Pipeline du meilleur modèle sauvegardé
│
├── notebooks
│   ├── 01_eda_preparation.ipynb    <- Exploration et préparation des données
│   ├── 02_modelisation.ipynb       <- Modélisation et comparaison initiale
│   └── 03_mlflow_experimentations.ipynb
│                                    <- Expérimentations et suivi avec MLflow
│
├── reports
│   └── figures                     <- Graphiques et figures du rapport
│
├── references                      <- Documents et ressources de référence
│
├── tests
│   └── test_data.py                <- Tests issus de la structure du projet
│
├── tp_prediction_automobile        <- Package Python du projet
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling
│       ├── __init__.py
│       ├── predict.py
│       └── train.py
│
├── docs                            <- Documentation complémentaire du projet
├── environment.yml                 <- Configuration de l'environnement Conda
├── requirements.txt                <- Dépendances Python nécessaires
├── pyproject.toml                  <- Configuration du projet Python
├── Makefile                        <- Commandes utilitaires du projet
├── LICENSE
└── README.md                       <- Documentation principale du projet
```
## Dataset et préparation des données

Le projet utilise un dataset provenant d'un concessionnaire automobile contenant les caractéristiques de véhicules ainsi que leur prix.

Le dataset initial contient **544 véhicules**. La variable cible utilisée pour la prédiction est `PRICE`.

Les principales étapes de préparation ont été réalisées dans le notebook :

`notebooks/01_eda_preparation.ipynb`

Après le nettoyage et la préparation des données, le dataset utilisé pour la modélisation est sauvegardé dans :

`data/interim/vehicles_clean.csv`

Le dataset nettoyé contient **543 observations**.

### Variables utilisées

Les variables numériques utilisées pour la modélisation sont :

- `YEAR`
- `KM`
- `BODYDOORS`
- `PASSENGERS`

Les variables catégorielles sont :

- `MAKE`
- `MODEL`
- `BODYDESCF`
- `DRIVETRAIN`
- `FUEL`
- `TRANSDESCF`
- `COLORF`

La variable cible est :

- `PRICE`

### Prétraitement pour la modélisation

Les données sont séparées en données d'entraînement et de test selon une proportion de **80 % / 20 %**, avec `random_state=42`.

Les variables catégorielles sont transformées avec `OneHotEncoder(handle_unknown="ignore")`.

Les variables numériques sont conservées telles quelles.

Ces transformations sont regroupées dans un `ColumnTransformer`, puis intégrées avec le modèle dans un pipeline `scikit-learn`.

Cette approche permet de sauvegarder ensemble le prétraitement et le modèle afin d'utiliser exactement le même processus lors des prédictions effectuées par l'API.

## Modèles et expérimentations MLflow

Les expérimentations sont réalisées dans le notebook :

`notebooks/03_mlflow_experimentations.ipynb`

MLflow est utilisé pour suivre les différentes expérimentations et enregistrer les paramètres, les métriques et les modèles entraînés.

L'expérience MLflow utilisée est :

`prediction_prix_automobile`

Les performances sont évaluées avec trois métriques de régression :

- **MAE (Mean Absolute Error)** : erreur absolue moyenne entre le prix réel et le prix prédit ;
- **RMSE (Root Mean Squared Error)** : mesure l'erreur de prédiction en pénalisant davantage les erreurs importantes ;
- **R² (coefficient de détermination)** : mesure la proportion de la variation du prix expliquée par le modèle.

Pour MAE et RMSE, une valeur plus faible indique une meilleure performance. Pour R², une valeur plus élevée est recherchée.

### Runs comparés

| Run MLflow | Modèle | Configuration | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| `linear_regression` | Linear Regression | Baseline | 2281.56 | 3336.21 | 0.7543 |
| `ridge_alpha_1` | Ridge | alpha = 1 | 3915.52 | 5000.92 | 0.4479 |
| `ridge_alpha_10` | Ridge | alpha = 10 | 3920.92 | 5005.06 | 0.4470 |
| `lasso_alpha_1` | Lasso | alpha = 1 | **1858.34** | **2888.81** | **0.8158** |
| `lasso_alpha_10` | Lasso | alpha = 10 | 2113.45 | 2894.33 | 0.8151 |
| `random_forest_100` | Random Forest | n_estimators = 100 | 2563.30 | 3574.42 | 0.7179 |
| `random_forest_200` | Random Forest | n_estimators = 200 | 2562.11 | 3552.18 | 0.7214 |

## Sélection du modèle final

Parmi les configurations testées, **Lasso avec `alpha=1`** obtient les meilleures performances sur le jeu de test utilisé pour cette expérimentation.

Il présente :

- la MAE la plus faible : **1858.34** ;
- le RMSE le plus faible : **2888.81** ;
- le R² le plus élevé : **0.8158**.

Ce modèle a donc été sélectionné comme modèle final du projet.

Le pipeline complet, incluant le prétraitement des données et le modèle Lasso, est sauvegardé dans :

`models/best_model.joblib`

Le choix du modèle repose sur les performances observées sur le jeu de test avec la séparation utilisée dans ce projet. Les résultats ne garantissent donc pas que Lasso sera systématiquement le meilleur modèle sur de nouvelles données.

## API FastAPI

Le modèle sélectionné est exposé à travers une API développée avec FastAPI.

Le code de l'API se trouve dans :

`api/main.py`

L'API charge le pipeline sauvegardé dans :

`models/best_model.joblib`

La route principale utilisée pour effectuer une prédiction est :

`POST /predict`

Elle reçoit les caractéristiques d'un véhicule, applique automatiquement le prétraitement contenu dans le pipeline, puis retourne le prix prédit.

Exemple simplifié de réponse :

```json
{
  "prediction": 36018.41,
  "model_name": "Lasso",
  "alpha": 1
}
```

Les informations sur le modèle sont récupérées à partir du pipeline chargé par l'API.

## Interface Streamlit

L'interface utilisateur est développée avec Streamlit.

Le code se trouve dans :

`app/streamlit_app.py`

L'utilisateur peut saisir les caractéristiques d'un véhicule dans un formulaire puis demander une estimation de son prix.

Streamlit n'effectue pas directement la prédiction. L'application envoie les données à FastAPI avec une requête HTTP `POST`.

Le processus de prédiction est donc :

**Utilisateur → Streamlit → requête HTTP → FastAPI → pipeline ML → prédiction → FastAPI → Streamlit**

Cette séparation permet de garder la logique de prédiction dans l'API et d'utiliser Streamlit uniquement comme interface utilisateur.

Après réception de la réponse de FastAPI, Streamlit affiche :

- le prix prédit ;
- le nom du modèle utilisé ;
- la valeur du paramètre `alpha`.


## Installation

Depuis la racine du projet, installer les dépendances Python :

```bash
pip install -r requirements.txt
```

Les principales dépendances du projet sont notamment :

- pandas
- scikit-learn
- MLflow
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Requests

## Lancement du projet

L'application utilise trois composants distincts : MLflow, FastAPI et Streamlit.

Il est recommandé de les lancer dans des terminaux séparés.

### 1. Lancer MLflow

Les expérimentations MLflow de ce projet sont stockées dans le dossier `notebooks`.

Depuis la racine du projet :

```bash
cd notebooks
mlflow ui
```

L'interface MLflow est ensuite accessible à l'adresse :

`http://127.0.0.1:5000`

MLflow permet de consulter les différents runs, leurs paramètres, leurs métriques et les modèles enregistrés pendant les expérimentations.

### 2. Lancer FastAPI

Dans un deuxième terminal, depuis la racine du projet :

```bash
uvicorn api.main:app --reload
```

L'API est accessible à l'adresse :

`http://127.0.0.1:8000`

La documentation interactive Swagger permet de tester la route `POST /predict` :

`http://127.0.0.1:8000/docs`

### 3. Lancer Streamlit

Dans un troisième terminal, depuis la racine du projet :

```bash
streamlit run app/streamlit_app.py
```

Streamlit démarre l'interface permettant de saisir les caractéristiques d'un véhicule et d'obtenir une estimation de son prix.

FastAPI doit être en cours d'exécution pour que Streamlit puisse effectuer une prédiction.

## Ordre de lancement recommandé

```text
Terminal 1
└── MLflow

Terminal 2
└── FastAPI
    └── charge models/best_model.joblib

Terminal 3
└── Streamlit
    └── envoie les données à FastAPI par HTTP
```

Pour effectuer une prédiction depuis l'interface, les composants indispensables sont **FastAPI et Streamlit**. MLflow est utilisé séparément pour consulter et présenter les expérimentations.

## Limites du projet

Ce projet démontre une démarche MLOps simple de bout en bout, mais certaines limites doivent être prises en compte.

- Le dataset est relativement petit, avec 543 observations après nettoyage.
- Les performances des modèles ont été évaluées sur une seule séparation train/test avec `random_state=42`.
- Les résultats obtenus peuvent donc varier avec un autre échantillonnage des données.
- Les variables numériques n'ont pas été standardisées. Cette étape pourrait notamment être étudiée pour les modèles de régularisation comme Ridge et Lasso.
- Les modèles et configurations testés représentent un nombre limité de possibilités.
- L'application FastAPI et l'interface Streamlit sont exécutées localement et ne sont pas déployées sur un serveur de production.

## Améliorations possibles

Plusieurs améliorations pourraient être apportées dans une version future du projet :

- utiliser davantage de données afin d'améliorer la représentativité du dataset ;
- utiliser la validation croisée pour obtenir une évaluation plus robuste des modèles ;
- évaluer l'effet de la standardisation des variables numériques sur Ridge et Lasso ;
- tester d'autres modèles ou effectuer une recherche plus systématique des hyperparamètres ;
- enrichir le suivi des expérimentations dans MLflow ;
- ajouter des tests automatisés pour l'API et le pipeline de prédiction ;
- conteneuriser les différents composants avec Docker ;
- déployer FastAPI et Streamlit dans un environnement accessible à distance.

## Conclusion

Ce projet met en œuvre un processus complet de Machine Learning pour la prédiction du prix d'un véhicule automobile.

Les différentes expérimentations suivies avec MLflow ont permis de comparer plusieurs modèles et configurations. Parmi les runs réalisés, le modèle Lasso avec `alpha=1` a obtenu les meilleures performances sur le jeu de test utilisé et a été sélectionné comme modèle final.

Le pipeline contenant le prétraitement et le modèle a ensuite été sauvegardé et intégré à une API FastAPI. Une interface Streamlit communique avec cette API par requête HTTP afin de permettre à un utilisateur d'effectuer une prédiction.

Le projet illustre ainsi le passage d'une expérimentation de Machine Learning à une petite application fonctionnelle, tout en conservant le suivi des expérimentations et une séparation claire entre le modèle, l'API et l'interface utilisateur.

--------

