# TP Prediction automobile

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Objectif du projet

Ce projet a pour objectif de prédire le prix d'un véhicule automobile à partir de ses caractéristiques.

Il s'agit d'un problème de régression supervisée. Le projet applique une démarche MLOps simple allant de la préparation des données jusqu'à l'utilisation du modèle dans une application.

La démarche comprend :

- la préparation et l'exploration des données ;
- l'entraînement et la comparaison de plusieurs familles de modèles (référence, linéaires, linéaires régularisés, arbres et ensembles) ;
- l'évaluation sur un jeu de test et par validation croisée à 5 plis ;
- le suivi des expérimentations finales avec MLflow ;
- la sélection et la sauvegarde du modèle champion ;
- l'exposition du modèle avec une API FastAPI ;
- la création d'une interface Streamlit communiquant avec l'API par requête HTTP ;
- la conteneurisation avec Docker et une intégration continue GitHub Actions.

## Démarche du projet

**Données → Préparation → Modélisation et comparaison → Expérimentations MLflow → Sélection du champion → FastAPI → Streamlit**

## Project Organization

```
├── .github/workflows/ci.yml        <- Intégration continue (tests + build des images Docker)
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
│   └── raw                         <- Données originales (vehicles.csv, vehicles_2.csv)
│
├── models
│   └── best_model.joblib           <- Pipeline du modèle champion (ElasticNetCV)
│
├── notebooks
│   ├── 01_eda_preparation.ipynb    <- Exploration et préparation des données
│   ├── 02_modelisation.ipynb       <- Modélisation et comparaison complète des modèles
│   ├── 03_mlflow_experimentations_final.ipynb
│   │                                <- Runs MLflow finaux et sauvegarde du champion
│   ├── backup_03_mlflow_experimentations.ipynb
│   │                                <- Version initiale (543 véhicules), conservée pour l'historique
│   └── mlflow.db, mlruns/          <- Suivi MLflow
│
├── reports
│   ├── rapport_final.md            <- Rapport du projet
│   └── figures                     <- Graphiques et captures
│
├── tests
│   └── test_data.py                <- Test de la route GET / de l'API
│
├── tp_prediction_automobile        <- Package Python du projet (structure du gabarit)
├── Dockerfile.api                  <- Image de l'API FastAPI
├── Dockerfile.streamlit            <- Image de l'interface Streamlit
├── docker-compose.yml              <- Lancement conjoint API + Streamlit
├── environment.yml                 <- Configuration de l'environnement Conda
├── requirements.txt                <- Dépendances Python nécessaires
├── pyproject.toml                  <- Configuration du projet Python
├── Makefile                        <- Commandes utilitaires du projet
├── LICENSE
└── README.md                       <- Documentation principale du projet
```

## Dataset et préparation des données

Le projet utilise des données provenant d'un concessionnaire automobile contenant les caractéristiques de véhicules ainsi que leur prix.

Deux fichiers bruts (`vehicles.csv` et `vehicles_2.csv`) sont fusionnés, puis dédoublonnés à l'aide de l'identifiant `ID` en conservant la version la plus récente de chaque véhicule. La fusion est triée de façon stable (date de modification, puis `ID`) et écrite dans `data/interim/vehicles_merged.csv` : les fichiers de `data/raw` ne sont plus modifiés et l'ordre des lignes, donc la séparation train/test, est reproductible. Le dataset consolidé contient **1 748 véhicules** et 106 variables brutes.

La variable cible utilisée pour la prédiction est `PRICE`.

Les principales étapes de préparation ont été réalisées dans le notebook `notebooks/01_eda_preparation.ipynb`. Le dataset utilisé pour la modélisation (**1 748 observations, 12 colonnes**) est sauvegardé dans `data/interim/vehicles_clean.csv`.

### Variables utilisées

Variables numériques : `YEAR`, `KM`, `BODYDOORS`, `PASSENGERS`.

Variables catégorielles : `MAKE`, `MODEL`, `BODYDESCF`, `DRIVETRAIN`, `FUEL`, `TRANSDESCF`, `COLORF`.

Variable cible : `PRICE`.

Les variables directement liées au prix (par exemple `LISTPRICE`, `COSTPRICE`, `WHOLESALEPRICE`, `basePrice`), les identifiants, les URL et les textes libres sont exclus afin d'éviter les fuites de données.

### Nettoyage

- les valeurs manquantes de `FUEL` et `DRIVETRAIN` sont remplacées par la catégorie `Inconnu` ;
- les cinq valeurs `BODYDOORS = 0` sont remplacées par la médiane des valeurs valides (4) ;
- les quatre valeurs `PASSENGERS = 0` sont conservées telles quelles ;
- les valeurs extrêmes de prix et de kilométrage sont conservées, car elles correspondent à des véhicules réels.

Quelques constats de l'analyse exploratoire :

- la distribution du prix est asymétrique (médiane 21 995 $, moyenne d'environ 23 600 $, maximum 119 900 $) ;
- `YEAR` (corrélation d'environ 0,56) et `KM` (environ -0,29) sont les variables numériques les plus liées au prix ;
- `MODEL` contient 222 catégories, dont 147 apparaissent cinq fois ou moins (environ 17,5 % des véhicules).

### Prétraitement pour la modélisation

Les données sont séparées en données d'entraînement et de test selon une proportion de **80 % / 20 %**, avec `random_state=42`, soit **1 398** véhicules pour l'entraînement et **350** pour le test.

Les variables catégorielles sont transformées avec `OneHotEncoder(handle_unknown="ignore")`, ce qui produit 288 colonnes après encodage.

Deux prétraitements sont utilisés :

- `preprocessor` : variables numériques conservées telles quelles (LinearRegression, arbres, RandomForest, XGBoost) ;
- `preprocessor_scaled` : variables numériques standardisées avec `StandardScaler` (Ridge, Lasso, ElasticNet).

Ces transformations sont regroupées dans un `ColumnTransformer`, puis intégrées avec le modèle dans un pipeline `scikit-learn`. Le pipeline est ajusté sur les seules données d'entraînement (et réajusté dans chaque pli de validation croisée), puis sauvegardé tel quel pour être utilisé par l'API.

## Modèles et évaluation

La comparaison complète est réalisée dans `notebooks/02_modelisation.ipynb`.

Les performances sont évaluées avec :

- **MAE (Mean Absolute Error)** : erreur absolue moyenne entre le prix réel et le prix prédit, en dollars ;
- **RMSE (Root Mean Squared Error)** : erreur en dollars qui pénalise davantage les erreurs importantes ;
- **R² (coefficient de détermination)** : part de la variance du prix expliquée par le modèle par rapport à une prédiction constante égale à la moyenne ;
- **R² moyen en validation croisée** à 5 plis, calculé sur l'ensemble des 1 748 véhicules.

Pour MAE et RMSE, une valeur plus faible indique une meilleure performance. Pour R², une valeur plus élevée est recherchée.

### Résultats sur le jeu de test et en validation croisée

| Modèle | Configuration | MAE ($) | RMSE ($) | R² test | R² CV moyen |
|---|---|---:|---:|---:|---:|
| DummyRegressor | moyenne | 5 741 | 7 872 | -0,016 | — |
| DummyRegressor | médiane | 5 543 | 7 814 | -0,002 | — |
| LinearRegression | — | 2 274 | 4 243 | 0,705 | 0,723 |
| Ridge | alpha = 1 | 1 832 | 3 777 | 0,766 | 0,769 |
| RidgeCV | alpha choisi = 1 | 1 832 | 3 777 | 0,766 | 0,764 |
| Lasso | alpha = 1 | **1 726** | 3 848 | 0,757 | 0,755 |
| **ElasticNetCV** | alpha = 0,001 ; l1_ratio = 0,5 | 1 800 | **3 751** | **0,769** | **0,772** |
| DecisionTree | min_samples_leaf = 2 | 2 582 | 4 653 | 0,645 | 0,455 |
| RandomForest | 100 arbres | 1 973 | 4 537 | 0,662 | 0,711 |
| XGBoost | 100 arbres, learning_rate 0,1, profondeur 3 | 2 109 | 4 503 | 0,668 | 0,769 |

Autres variantes testées sur le jeu de test : Ridge alpha=10 (R² 0,726), Lasso alpha=10 (R² 0,712), ElasticNet alpha=1 / l1_ratio=0,5 (R² 0,494), RandomForest 200 arbres (R² 0,655), RandomForest profondeur 10 (R² 0,622), cinq autres configurations XGBoost et plusieurs profondeurs d'arbre de décision.

## Expérimentations MLflow

Les runs finaux sont enregistrés dans le notebook `notebooks/03_mlflow_experimentations_final.ipynb`, dans l'expérience MLflow **`prediction_prix_automobile_final`**.

Ce notebook reprend les cinq modèles retenus à l'issue du notebook 02. Pour chaque run, MLflow enregistre la taille du test, le `random_state`, le nombre de lignes, les hyperparamètres (y compris `alpha` et `l1_ratio` choisis automatiquement), les métriques `mae`, `rmse`, `r2_test`, `r2_cv_mean` et le pipeline entraîné.

| Run MLflow | MAE ($) | RMSE ($) | R² test | R² CV moyen |
|---|---:|---:|---:|---:|
| `final_elasticnetcv` | 1 800,09 | 3 751,06 | 0,7692 | 0,7716 |
| `final_ridgecv` | 1 831,63 | 3 777,28 | 0,7660 | 0,7636 |
| `final_lasso_alpha_1` | 1 725,97 | 3 847,80 | 0,7572 | 0,7549 |
| `final_linearregression` | 2 274,19 | 4 242,58 | 0,7048 | 0,7227 |
| `final_xgboost` | 2 109,02 | 4 502,50 | 0,6675 | 0,7687 |

Les modèles écartés dans le notebook 02 (Dummy, arbre de décision, RandomForest, variantes de Ridge, Lasso, ElasticNet et XGBoost) ne sont pas journalisés dans MLflow ; leurs résultats sont conservés dans le notebook 02.

25 runs d'exécutions antérieures du notebook final ont été supprimés de l'expérience : trois séries identiques entre elles (dont deux runs XGBoost en échec lors de l'enregistrement du modèle) et deux séries obtenues avant que l'ordre des lignes soit rendu reproductible, sur d'autres découpages train/test. Seuls les 5 runs du 14 octobre (version reproductible) sont actifs. L'expérience de la version initiale (`prediction_prix_automobile`, 543 véhicules) n'est plus présente dans `mlflow.db`.

## Sélection du modèle final

**ElasticNetCV** (`alpha=0.001`, `l1_ratio=0.5`, choisis par validation croisée interne à 5 plis) est retenu comme modèle champion.

Parmi les cinq runs finaux, il obtient :

- le R² test le plus élevé : **0,769** ;
- le RMSE le plus faible : **3 751 $** ;
- la deuxième meilleure MAE : **1 800 $** (derrière Lasso alpha=1, 1 726 $) ;
- le R² moyen en validation croisée le plus élevé : **0,772** (XGBoost : 0,769).

Il est retenu pour son meilleur compromis entre performance sur le jeu de test, stabilité en validation croisée et simplicité. Les écarts avec Ridge et Lasso restent toutefois très faibles et inférieurs à la variabilité observée entre les plis de validation croisée.

Le pipeline complet (prétraitement standardisé + ElasticNetCV) est sauvegardé dans `models/best_model.joblib`.

## API FastAPI

Le modèle sélectionné est exposé à travers une API développée avec FastAPI (`api/main.py`).

L'API charge le pipeline sauvegardé dans `models/best_model.joblib`. Les routes disponibles sont :

- `GET /` : message d'accueil et nom du modèle champion ;
- `POST /predict` : reçoit les caractéristiques d'un véhicule, applique automatiquement le prétraitement contenu dans le pipeline, puis retourne le prix prédit.

Exemple de requête :

```json
{
  "YEAR": 2022,
  "KM": 55098,
  "BODYDOORS": 5,
  "PASSENGERS": 7,
  "MAKE": "Jeep",
  "MODEL": "Grand Cherokee L",
  "BODYDESCF": "VUS",
  "DRIVETRAIN": "4 roues motrices",
  "FUEL": "Essence",
  "TRANSDESCF": "Automatique",
  "COLORF": "Charbon"
}
```

Réponse obtenue avec le modèle champion actuel :

```json
{
  "prediction": 32763.83,
  "model_name": "ElasticNetCV",
  "alpha": 0.001
}
```

Le nom du modèle et la valeur de `alpha` sont récupérés à partir du pipeline chargé par l'API.

## Interface Streamlit

L'interface utilisateur est développée avec Streamlit (`app/streamlit_app.py`).

L'utilisateur saisit les caractéristiques d'un véhicule dans un formulaire puis demande une estimation de son prix. Streamlit n'effectue pas directement la prédiction : l'application envoie les données à FastAPI avec une requête HTTP `POST`. L'adresse de l'API est configurable avec la variable d'environnement `API_URL` (par défaut `http://127.0.0.1:8000`).

**Utilisateur → Streamlit → requête HTTP → FastAPI → pipeline ML → prédiction → FastAPI → Streamlit**

Après réception de la réponse, Streamlit affiche le prix prédit, le nom du modèle utilisé et la valeur du paramètre `alpha`.

## Installation

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

Principales dépendances : pandas, scikit-learn, XGBoost, MLflow, Joblib, FastAPI, Uvicorn, Streamlit, Requests, pytest.

## Lancement du projet

### 1. Lancer MLflow

Les expérimentations MLflow sont stockées dans le dossier `notebooks` :

```bash
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface MLflow est accessible à l'adresse `http://127.0.0.1:5000`.

### 2. Lancer FastAPI

```bash
uvicorn api.main:app --reload
```

L'API est accessible à l'adresse `http://127.0.0.1:8000` et sa documentation interactive à `http://127.0.0.1:8000/docs`.

### 3. Lancer Streamlit

```bash
streamlit run app/streamlit_app.py
```

FastAPI doit être en cours d'exécution pour que Streamlit puisse effectuer une prédiction.

### Avec Docker

```bash
docker compose up --build
```

L'API est alors disponible sur le port 8000 et Streamlit sur le port 8501.

### Tests et intégration continue

```bash
python -m pytest
```

Le workflow GitHub Actions (`.github/workflows/ci.yml`) exécute les tests puis construit les images Docker de l'API et de Streamlit à chaque push ou pull request sur `main`.

## Limites du projet

- Le dataset reste de taille modeste (1 748 véhicules) et provient d'un seul concessionnaire ; il contient peu de véhicules haut de gamme.
- Les modèles sont comparés sur une seule séparation train/test (`random_state=42`). Il n'y a pas de jeu de validation distinct : le jeu de test a servi à comparer les variantes et à choisir le champion, ce qui rend son score légèrement optimiste.
- La validation croisée externe porte sur l'ensemble des données (jeu de test inclus), utilise des plis sans mélange et n'est calculée qu'avec le R².
- Les scores dépendent des véhicules tirés pour le test : deux exécutions antérieures, faites avant que l'ordre des lignes soit rendu reproductible, donnaient pour ElasticNetCV une MAE de 1 776 $ et 1 821 $ (1 800 $ aujourd'hui).
- Les écarts entre les modèles linéaires régularisés sont faibles ; aucun test statistique n'a été réalisé.
- Sur la matrice creuse produite par l'encodage, `LinearRegression` utilise un solveur itératif (`lsqr`) qui s'arrête avant le minimum des moindres carrés : ses scores sont sous-estimés et varient légèrement d'une exécution à l'autre. Une vérification réalisée hors notebook et hors MLflow (résolution exacte sur matrice dense) donne une MAE d'environ 1 690 $ et un R² moyen en validation croisée d'environ 0,765, au niveau des modèles régularisés ; ce résultat reste à intégrer au notebook.
- Les modèles commerciaux rares ou absents de l'entraînement (catégorie ignorée par l'encodage) produisent les erreurs les plus importantes.
- La médiane utilisée pour corriger `BODYDOORS` est calculée avant la séparation train/test (effet négligeable).
- L'application est exécutée localement ou via Docker, sans déploiement sur un serveur de production.

## Améliorations possibles

- utiliser un jeu de validation distinct ou une validation croisée imbriquée afin de conserver un jeu de test réellement final ;
- utiliser une validation croisée avec mélange et calculer également la MAE et le RMSE par pli ;
- réentraîner la régression linéaire avec un solveur exact (matrice dense) et l'enregistrer dans MLflow ;
- étendre la grille de `alpha` d'ElasticNetCV sous 0,001 (valeur retenue à la limite de la grille) ;
- regrouper les modèles rares ou tester une transformation logarithmique du prix ;
- utiliser `GridSearchCV` ou `RandomizedSearchCV` pour régler les modèles à base d'arbres ;
- ajouter des tests automatisés pour la route `/predict` ;
- déployer FastAPI et Streamlit dans un environnement accessible à distance.

## Conclusion

Ce projet met en œuvre un processus complet de Machine Learning pour la prédiction du prix d'un véhicule automobile.

La comparaison de plusieurs familles de modèles montre que tous les modèles dépassent nettement la référence naïve, que les modèles linéaires régularisés obtiennent les meilleures performances sur le jeu de test et que les modèles à base d'arbres n'apportent pas de gain mesurable sur ce jeu de données. **ElasticNetCV** a été retenu comme meilleur compromis.

Le pipeline contenant le prétraitement et le modèle a été sauvegardé, intégré à une API FastAPI et utilisé par une interface Streamlit, avec une conteneurisation Docker et une intégration continue.

--------
