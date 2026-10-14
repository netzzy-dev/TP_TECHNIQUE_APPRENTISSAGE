# Rapport final — Prédiction du prix d'un véhicule automobile

## 1. Présentation du problème choisi

Dans le secteur automobile, le prix d'un véhicule peut varier en fonction de plusieurs caractéristiques telles que son année, son kilométrage, sa marque, son modèle, son type de carrosserie ou encore son groupe motopropulseur.

L'objectif de ce projet est de développer un modèle de Machine Learning capable d'estimer le prix d'un véhicule à partir de ces caractéristiques.

Le projet ne se limite pas à l'entraînement d'un modèle. Il compare plusieurs familles de modèles selon un protocole commun, suit les expérimentations finales avec MLflow, sélectionne un modèle champion et le rend utilisable à travers une API FastAPI et une interface Streamlit.

La démarche générale du projet est la suivante :

**Données → Préparation → Modélisation et comparaison → Expérimentations MLflow → Sélection du modèle → FastAPI → Streamlit**

## 2. Description du dataset

Les données proviennent d'un concessionnaire automobile. Elles contiennent différentes caractéristiques descriptives de véhicules ainsi que leur prix.

Deux fichiers bruts (`vehicles.csv` et `vehicles_2.csv`) ont été fusionnés puis dédoublonnés à l'aide de l'identifiant `ID`, en conservant la version la plus récente de chaque véhicule. La fusion est triée de façon stable (date de modification, puis `ID`) et écrite dans `data/interim/vehicles_merged.csv`, afin que l'ordre des lignes, et donc la séparation train/test, soit reproductible. Le dataset consolidé contient **1 748 véhicules** et 106 variables brutes. Aucune observation n'a été supprimée lors du nettoyage : **1 748 observations** sont utilisées pour la modélisation.

La variable à prédire est :

- `PRICE` : prix affiché du véhicule.

Les variables explicatives retenues sont :

### Variables numériques

- `YEAR` : année du véhicule ;
- `KM` : kilométrage ;
- `BODYDOORS` : nombre de portes ;
- `PASSENGERS` : nombre de passagers.

### Variables catégorielles

- `MAKE` : marque du véhicule ;
- `MODEL` : modèle du véhicule ;
- `BODYDESCF` : type de carrosserie ;
- `DRIVETRAIN` : type de transmission aux roues ;
- `FUEL` : type de carburant ;
- `TRANSDESCF` : type de transmission ;
- `COLORF` : couleur du véhicule.

Le dataset préparé est sauvegardé dans `data/interim/vehicles_clean.csv`.

### Principaux constats de l'analyse exploratoire

- La distribution du prix est asymétrique vers les prix élevés : médiane de 21 995 $, moyenne d'environ 23 600 $, maximum de 119 900 $. Environ 82 % des véhicules ont un prix compris entre 15 000 $ et 35 000 $.
- Parmi les variables numériques, `YEAR` présente la corrélation la plus forte avec le prix (environ 0,56), suivie de `KM` (environ -0,29). `BODYDOORS` n'a presque pas de relation linéaire avec le prix.
- Les prix médians diffèrent selon la marque et le type de carrosserie.
- `MODEL` présente une forte cardinalité : 222 modèles différents, dont 147 apparaissent cinq fois ou moins (306 véhicules, environ 17,5 % du dataset).

Ces constats ont orienté la modélisation : la forte cardinalité de `MODEL` produit un grand nombre de colonnes après encodage, ce qui motive l'utilisation de la régularisation, et la présence de véhicules très chers rend le RMSE sensible à quelques observations.

## 3. Type de problème

Le projet correspond à un problème de **régression supervisée**.

Il s'agit d'un apprentissage supervisé puisque le dataset contient, pour chaque véhicule, les caractéristiques utilisées comme variables explicatives ainsi que le prix réel `PRICE` utilisé comme variable cible.

Il s'agit plus précisément d'un problème de régression puisque la valeur à prédire est une valeur numérique continue : le prix d'un véhicule.

## 4. Préparation des données

La préparation des données a été réalisée dans le notebook `notebooks/01_eda_preparation.ipynb`.

### Sélection des variables

Seules 11 variables explicatives ont été conservées parmi les 106 variables brutes. Les variables directement liées au prix (par exemple `LISTPRICE`, `COSTPRICE`, `WHOLESALEPRICE`, `basePrice` ou les informations de financement), les identifiants, les URL et les textes libres ont été exclus afin d'éviter les fuites de données.

### Nettoyage

- Les valeurs manquantes de `FUEL` (10) et `DRIVETRAIN` (1) sont remplacées par la catégorie `Inconnu`.
- Les cinq valeurs `BODYDOORS = 0` sont remplacées par la médiane des valeurs valides (4). Cette médiane est calculée sur l'ensemble du dataset, avant la séparation train/test ; l'effet est négligeable.
- Les quatre valeurs `PASSENGERS = 0`, correspondant principalement à des véhicules commerciaux, sont conservées telles quelles.
- Les valeurs extrêmes de prix et de kilométrage sont conservées puisqu'elles correspondent à des véhicules réels.

### Séparation des données

Les données sont séparées en deux ensembles avec `random_state=42` :

- **80 %** pour l'entraînement, soit **1 398 observations** ;
- **20 %** pour le test, soit **350 observations**.

### Transformation des variables

Les variables catégorielles sont transformées avec `OneHotEncoder(handle_unknown="ignore")`. Après encodage, les données d'entraînement comptent **288 colonnes**. L'option `handle_unknown="ignore"` évite une erreur lorsqu'une catégorie inconnue est rencontrée lors d'une prédiction.

Deux prétraitements sont utilisés selon les modèles :

- variables numériques conservées telles quelles pour la régression linéaire, l'arbre de décision, RandomForest et XGBoost ;
- variables numériques standardisées avec `StandardScaler` pour Ridge, Lasso et ElasticNet, afin que la pénalité de régularisation traite les variables sur une échelle comparable.

### Pipeline de prétraitement

Un `ColumnTransformer` applique les transformations appropriées selon le type de variable, puis il est combiné avec le modèle dans un pipeline `scikit-learn`.

Le pipeline est ajusté sur les seules données d'entraînement et réajusté dans chaque pli de validation croisée, ce qui évite les fuites de données lors de l'encodage et de la standardisation. Le pipeline complet est ensuite sauvegardé et réutilisé par FastAPI.

## 5. Protocole d'évaluation

### Métriques

- **MAE (Mean Absolute Error)** : erreur absolue moyenne entre les prix réels et prédits, en dollars.
- **RMSE (Root Mean Squared Error)** : erreur en dollars qui pénalise davantage les erreurs importantes ; elle est donc sensible aux véhicules très chers.
- **R² (coefficient de détermination)** : part de la variance du prix expliquée par le modèle par rapport à une prédiction constante égale à la moyenne. Un R² de 0 correspond à cette prédiction constante et un R² peut être négatif. Il ne s'agit pas d'un pourcentage de prédictions correctes.

### Validation

- Tous les modèles sont entraînés et évalués sur la même séparation train/test.
- Les principaux modèles sont également évalués par **validation croisée à 5 plis** (`cross_val_score`, scoring R²) sur l'ensemble des 1 748 véhicules. On rapporte la moyenne et la dispersion des scores.
- Pour RidgeCV et ElasticNetCV, une **validation croisée interne à 5 plis** sur les données d'entraînement sélectionne automatiquement les hyperparamètres.

### Limites du protocole

- Il n'y a pas de jeu de validation distinct : le jeu de test a été utilisé pour comparer les variantes (profondeur des arbres, configurations XGBoost) et pour choisir le champion. Son score est donc une estimation légèrement optimiste.
- La validation croisée externe inclut les observations du jeu de test, utilise des plis sans mélange et n'est calculée qu'avec le R².
- Les scores dépendent des véhicules tirés pour le test. Avant la correction de l'ordre des lignes, deux exécutions précédentes (runs MLflow supprimés) avaient produit d'autres découpages : la MAE d'ElasticNetCV y valait 1 776 $ et 1 821 $, contre 1 800 $ aujourd'hui. Cet écart est du même ordre que celui qui sépare les meilleurs modèles.

## 6. Modèles testés

La comparaison complète est réalisée dans `notebooks/02_modelisation.ipynb`.

### DummyRegressor

Modèle de référence minimale qui prédit toujours la moyenne (ou la médiane) des prix d'entraînement, sans utiliser les caractéristiques. Il permet de vérifier que les autres modèles apprennent réellement une relation.

### Linear Regression

La régression linéaire représente le prix comme une combinaison linéaire des variables. Elle constitue le premier véritable modèle prédictif.

### Ridge, Lasso et ElasticNet

Ces modèles ajoutent une pénalité sur la taille des coefficients afin de limiter la variance, particulièrement utile avec 288 colonnes pour 1 398 observations :

- **Ridge** (pénalité L2) réduit tous les coefficients sans les annuler. Configurations : `alpha = 1`, `alpha = 10`, puis RidgeCV avec `alpha` ∈ {0,01 ; 0,1 ; 1 ; 10 ; 100} → `alpha = 1` retenu.
- **Lasso** (pénalité L1) peut annuler certains coefficients. Configurations : `alpha = 1` et `alpha = 10`, avec `max_iter = 10000`.
- **ElasticNet** combine les deux pénalités. Configuration initiale `alpha = 1`, `l1_ratio = 0.5`, puis ElasticNetCV avec `alpha` ∈ {0,001 ; 0,01 ; 0,1 ; 1 ; 10} et `l1_ratio` ∈ {0,1 ; 0,5 ; 0,9} → `alpha = 0.001`, `l1_ratio = 0.5` retenus.

### DecisionTreeRegressor

Un arbre de décision peut représenter des relations non linéaires et des interactions. Plusieurs valeurs de `max_depth` (3, 5, 10, aucune limite) et de `min_samples_leaf` (1, 2, 5, 10, 20) ont été testées.

### RandomForestRegressor (bagging)

Ensemble d'arbres entraînés sur des échantillons bootstrap dont les prédictions sont moyennées, ce qui réduit la variance. Configurations : 100 arbres, 200 arbres, 100 arbres avec `max_depth = 10`.

### XGBoost (boosting)

Ensemble d'arbres peu profonds construits de manière séquentielle, chaque arbre corrigeant une partie des erreurs des précédents. Configuration de référence : 100 arbres, `learning_rate = 0.1`, `max_depth = 3`. Cinq autres configurations ont été testées sans amélioration sur le jeu de test.

## 7. Résultats et analyse

| Modèle | Configuration | MAE ($) | RMSE ($) | R² test | R² CV (moyenne ± écart-type) |
|---|---|---:|---:|---:|---:|
| DummyRegressor | moyenne | 5 741 | 7 872 | -0,016 | — |
| DummyRegressor | médiane | 5 543 | 7 814 | -0,002 | — |
| LinearRegression | — | 2 274 | 4 243 | 0,705 | 0,723 ± 0,110 |
| Ridge | alpha = 1 | 1 832 | 3 777 | 0,766 | 0,769 ± 0,108 |
| RidgeCV | alpha choisi = 1 | 1 832 | 3 777 | 0,766 | 0,764 ± 0,103 |
| Lasso | alpha = 1 | **1 726** | 3 848 | 0,757 | 0,755 ± 0,100 |
| **ElasticNetCV** | alpha = 0,001 ; l1_ratio = 0,5 | 1 800 | **3 751** | **0,769** | **0,772 ± 0,108** |
| DecisionTree | min_samples_leaf = 2 | 2 582 | 4 653 | 0,645 | 0,455 ± 0,208 |
| RandomForest | 100 arbres | 1 973 | 4 537 | 0,662 | 0,711 ± 0,102 |
| XGBoost | 100 arbres, lr 0,1, profondeur 3 | 2 109 | 4 503 | 0,668 | 0,769 ± 0,054 |

Autres variantes, évaluées sur le jeu de test uniquement : Ridge `alpha=10` (MAE 2 137 $, R² 0,726), Lasso `alpha=10` (MAE 2 136 $, R² 0,712), ElasticNet `alpha=1`, `l1_ratio=0.5` (MAE 3 424 $, R² 0,494), RandomForest 200 arbres (R² 0,655), RandomForest `max_depth=10` (R² 0,622).

### Analyse

- **Tous les modèles dépassent nettement la référence** : la MAE passe d'environ 5 700 $ à 1 700–2 600 $.
- **Régularisation** : avec `alpha = 1`, Ridge et Lasso réduisent nettement la MAE par rapport à la régression linéaire. Une régularisation trop forte dégrade les résultats (Ridge et Lasso `alpha=10`, ElasticNet `alpha=1`) : c'est un sous-apprentissage. Comme la régression linéaire n'utilise pas de standardisation, l'écart observé combine l'effet de la régularisation et celui de la standardisation.
- **Arbre de décision** : sans contrainte, il obtient un R² de 1,0 à l'entraînement et d'environ 0,51 au test, ce qui montre un fort surapprentissage. `min_samples_leaf = 2` améliore le R² test (environ 0,64), mais le modèle reste très instable en validation croisée.
- **Ensembles** : RandomForest n'améliore pas les modèles linéaires et le nombre d'arbres a peu d'effet. XGBoost obtient un R² moyen en validation croisée presque égal à celui d'ElasticNetCV (0,769 contre 0,772), avec la plus faible dispersion entre les plis, mais un R² test nettement inférieur (environ 0,67) et un écart entraînement/test important (R² entraînement d'environ 0,90).
- **Écarts entre les meilleurs modèles** : les modèles linéaires régularisés sont séparés de moins de 0,02 de R² moyen en validation croisée, alors que l'écart-type entre les plis est d'environ 0,10. Ces écarts ne permettent pas de conclure à une supériorité significative d'un modèle.

## 8. Expérimentations MLflow

Les runs finaux sont enregistrés dans `notebooks/03_mlflow_experimentations_final.ipynb`, dans l'expérience MLflow **`prediction_prix_automobile_final`** (base `notebooks/mlflow.db`).

Ce notebook reprend les cinq modèles retenus à l'issue du notebook 02 : LinearRegression, RidgeCV, Lasso (`alpha=1`), ElasticNetCV et XGBoost. Tous utilisent la même séparation train/test, les mêmes métriques et une validation croisée externe à 5 plis.

Pour chaque run, MLflow enregistre :

- les paramètres communs : `test_size`, `random_state_split`, `dataset_rows` ;
- les hyperparamètres du modèle, ainsi que `selected_alpha` et `selected_l1_ratio` pour les modèles sélectionnés par validation croisée ;
- les métriques `mae`, `rmse`, `r2_test` et `r2_cv_mean` ;
- le pipeline complet entraîné.

| Run MLflow | MAE ($) | RMSE ($) | R² test | R² CV moyen |
|---|---:|---:|---:|---:|
| `final_elasticnetcv` | 1 800,09 | 3 751,06 | 0,7692 | 0,7716 |
| `final_ridgecv` | 1 831,63 | 3 777,28 | 0,7660 | 0,7636 |
| `final_lasso_alpha_1` | 1 725,97 | 3 847,80 | 0,7572 | 0,7549 |
| `final_linearregression` | 2 274,19 | 4 242,58 | 0,7048 | 0,7227 |
| `final_xgboost` | 2 109,02 | 4 502,50 | 0,6675 | 0,7687 |

### Expériences non présentes dans MLflow

- **Modèles écartés dans le notebook 02** : Dummy, arbre de décision, RandomForest et variantes de Ridge, Lasso, ElasticNet et XGBoost ne sont pas journalisés. Le notebook 03 final est volontairement limité aux modèles retenus ; les résultats de ces modèles restent disponibles dans le notebook 02.
- **Runs supprimés** : 25 runs d'exécutions antérieures du notebook final ont été supprimés. Trois séries étaient identiques entre elles (deux runs XGBoost y avaient échoué lors de l'enregistrement du modèle) ; deux autres séries avaient été obtenues avant que l'ordre des lignes soit rendu reproductible, sur d'autres découpages train/test. Elles ne sont pas mélangées à la comparaison actuelle, mais montrent la sensibilité des scores au découpage.
- **Version initiale** : l'expérience `prediction_prix_automobile`, réalisée sur un dataset de 543 véhicules sans standardisation (notebook `backup_03_mlflow_experimentations.ipynb`), n'est plus présente dans `mlflow.db`. Ses résultats ne sont pas comparables aux résultats actuels.

> Les captures `figures/mlflow_runs.png` et `figures/mlflow_comparaison.png` correspondent à la version initiale du projet et doivent être régénérées à partir de l'expérience `prediction_prix_automobile_final`.

## 9. Choix du meilleur modèle

Le modèle **ElasticNetCV** (`alpha = 0.001`, `l1_ratio = 0.5`) a été sélectionné comme modèle champion.

Parmi les cinq runs finaux, il obtient :

- le R² test le plus élevé : **0,769** ;
- le RMSE le plus faible : **3 751 $** ;
- la deuxième meilleure MAE : **1 800 $**, derrière Lasso `alpha=1` (1 726 $) ;
- le R² moyen en validation croisée le plus élevé : **0,772**, devant XGBoost (0,769).

La MAE de **1 800 $** est la moyenne des erreurs absolues : sur le jeu de test, le prix prédit s'écarte en moyenne d'environ 1 800 $ du prix réel.

Le R² de **0,769** indique que le modèle réduit d'environ 77 % l'erreur quadratique par rapport à une prédiction constante égale au prix moyen. Il ne s'agit pas d'un pourcentage de prédictions correctes.

ElasticNetCV a été retenu pour son meilleur compromis entre performance sur le jeu de test, stabilité en validation croisée et simplicité : c'est un modèle linéaire, interprétable, dont les hyperparamètres ont été choisis par validation croisée interne. XGBoost, malgré un R² moyen en validation croisée presque égal, obtient des résultats nettement plus faibles sur le jeu de test.

Le pipeline complet d'ElasticNetCV, comprenant le prétraitement et le modèle, est sauvegardé dans `models/best_model.joblib`.

### Précaution d'interprétation

Le choix repose sur une seule séparation train/test et sur un jeu de test également utilisé pour la sélection. Les écarts avec Ridge et Lasso sont faibles et inférieurs à la variabilité observée entre les plis de validation croisée. ElasticNetCV est donc le meilleur compromis **parmi les configurations testées**, sans que sa supériorité soit démontrée statistiquement.

## 10. Description de l'API FastAPI

Une API a été développée avec **FastAPI** afin de rendre le modèle de prédiction accessible indépendamment de l'interface utilisateur. Le code se trouve dans `api/main.py`.

### Chargement du modèle

Au démarrage, l'API charge le pipeline sauvegardé dans `models/best_model.joblib`, qui contient le prétraitement des variables et le modèle ElasticNetCV.

### Routes

- `GET /` : retourne un message d'accueil et le nom du modèle champion.
- `POST /predict` : reçoit les caractéristiques d'un véhicule au format JSON et retourne le prix prédit, le nom du modèle et la valeur de `alpha`.

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

> La capture `figures/fastapi_predict.png` a été réalisée avec la version initiale du modèle (Lasso) et doit être régénérée.

## 11. Description de l'interface Streamlit

Une interface utilisateur a été développée avec **Streamlit** (`app/streamlit_app.py`) afin de permettre d'utiliser le modèle sans envoyer manuellement une requête à l'API.

### Saisie des caractéristiques

L'utilisateur saisit l'année, le kilométrage, le nombre de portes, le nombre de passagers, la marque, le modèle, le type de carrosserie, le groupe motopropulseur, le carburant, la transmission et la couleur, puis demande une estimation du prix.

### Communication avec FastAPI

Streamlit n'effectue pas directement la prédiction. L'application transmet les données à la route `POST /predict` de FastAPI par une requête HTTP. L'adresse de l'API est configurable avec la variable d'environnement `API_URL`.

**Utilisateur → Streamlit → requête HTTP → FastAPI → pipeline ML → prédiction → FastAPI → Streamlit**

### Affichage du résultat

Après réception de la réponse, l'interface affiche le prix prédit, le nom du modèle utilisé et la valeur du paramètre `alpha`. Pour le véhicule de l'exemple ci-dessus, le prix prédit par le modèle champion est d'environ **32 763,83 $**.

> La capture `figures/streamlit_prediction.png` a été réalisée avec la version initiale du modèle (Lasso) et doit être régénérée.

### Conteneurisation et intégration continue

Les deux composants disposent d'un Dockerfile (`Dockerfile.api`, `Dockerfile.streamlit`) et peuvent être lancés ensemble avec `docker compose up --build`. Un workflow GitHub Actions exécute les tests (`pytest`) puis construit les deux images Docker à chaque push ou pull request sur `main`.

## 12. Limites du projet

### Taille et représentativité du dataset

Le dataset contient 1 748 véhicules provenant d'un seul concessionnaire et peu de véhicules haut de gamme. Les prix utilisés sont des prix affichés et non des prix de vente.

### Protocole d'évaluation

- Les modèles sont comparés sur une seule séparation train/test (`random_state=42`).
- Il n'y a pas de jeu de validation distinct : le jeu de test a servi à la sélection des variantes et du champion.
- La validation croisée externe inclut le jeu de test, n'utilise pas de mélange et n'est calculée qu'avec le R².
- Aucun test statistique n'a été réalisé pour comparer les modèles.

### Comparaison entre régression linéaire et modèles régularisés

Sur la matrice creuse produite par l'encodage, `LinearRegression` utilise un solveur itératif (`lsqr`) qui s'arrête avant le minimum des moindres carrés (R² d'entraînement d'environ 0,86 au lieu de 0,95). Ses scores sont donc sous-estimés et varient légèrement d'une exécution à l'autre. Une vérification réalisée hors notebook et hors MLflow, par résolution exacte sur matrice dense, donne une MAE d'environ 1 690 $, un R² test d'environ 0,766 et un R² moyen en validation croisée d'environ 0,765, au niveau des modèles régularisés. L'apport de la régularisation n'est donc pas démontré sur ces données ; ce résultat reste à intégrer au notebook.

### Catégories rares

Les véhicules dont le modèle commercial est rare ou absent des données d'entraînement produisent les erreurs les plus importantes. Une catégorie inconnue est ignorée par l'encodage, ce qui empêche le modèle d'utiliser l'information correspondante.

### Hyperparamètres

La valeur `alpha = 0.001` retenue par ElasticNetCV se situe à la limite inférieure de la grille testée. Les modèles à base d'arbres ont été réglés à partir du jeu de test plutôt que par validation croisée.

### Environnement

L'application est exécutée localement ou avec Docker. Elle ne comprend pas encore les éléments nécessaires à un déploiement en production (hébergement, sécurité, supervision, mise à l'échelle).

## 13. Améliorations possibles

- Utiliser un jeu de validation distinct ou une validation croisée imbriquée afin de conserver un jeu de test réellement final.
- Utiliser une validation croisée avec mélange et calculer également la MAE et le RMSE par pli.
- Réentraîner la régression linéaire avec un solveur exact (matrice dense) et l'enregistrer dans MLflow.
- Étendre la grille de `alpha` d'ElasticNetCV sous 0,001.
- Regrouper les modèles rares ou tester une transformation logarithmique du prix pour réduire l'influence des véhicules très chers.
- Régler les modèles à base d'arbres avec `GridSearchCV` ou `RandomizedSearchCV`.
- Ajouter des tests automatisés pour la route `/predict`.
- Déployer FastAPI et Streamlit dans un environnement accessible à distance.
- Régénérer les captures MLflow, FastAPI et Streamlit avec la version actuelle du projet.

## 14. Conclusion

Ce projet avait pour objectif de développer un modèle de Machine Learning capable d'estimer le prix d'un véhicule automobile à partir de ses caractéristiques et de l'intégrer dans une démarche MLOps simple.

Après la préparation de 1 748 véhicules, plusieurs familles de modèles ont été comparées : une référence naïve, la régression linéaire, des modèles linéaires régularisés, un arbre de décision et deux méthodes d'ensemble. Tous les modèles dépassent nettement la référence. Les modèles linéaires régularisés obtiennent les meilleurs résultats sur le jeu de test, tandis que les modèles à base d'arbres n'apportent pas de gain mesurable sur ce jeu de données.

**ElasticNetCV** (`alpha = 0.001`, `l1_ratio = 0.5`) a été retenu comme meilleur compromis, avec une MAE de **1 800 $**, un RMSE de **3 751 $**, un R² test de **0,769** et un R² moyen en validation croisée de **0,772**. Les écarts avec Ridge et Lasso restent faibles.

Le pipeline contenant le prétraitement et le modèle a été sauvegardé dans `models/best_model.joblib`, puis intégré à une API FastAPI et utilisé par une interface Streamlit, avec une conteneurisation Docker et une intégration continue.

Les résultats restent liés au dataset et au protocole d'évaluation utilisés. Un jeu de validation distinct, une validation croisée plus complète et une comparaison avec une régression linéaire résolue exactement permettraient de renforcer les conclusions.
