# Conversion Rate Challenge - Prediction de Newsletter

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logoColor=white)
![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)

## Contexte du Projet

**datascienceweekly.org** est une newsletter curatee par des data scientists independants. L'equipe souhaite predire si un visiteur va s'abonner a la newsletter en fonction de son comportement de navigation et de ses caracteristiques demographiques.

Le projet s'inscrit dans un **challenge de Machine Learning** (format Kaggle) : construire le modele avec le meilleur **F1-Score** sur la prediction des conversions.

## Analyse Exploratoire (EDA)

### Donnees
- **284 580 lignes**, 6 colonnes, **aucune valeur manquante**
- 4 variables numeriques (`age`, `new_user`, `total_pages_visited`, `converted`)
- 2 variables categorielles (`country`, `source`)

### Desequilibre des classes
Le taux de conversion est de seulement **3.23%**. Ce fort desequilibre rend l'Accuracy inadaptee comme metrique : un modele qui predit toujours "pas converti" atteindrait 96.8% d'accuracy sans rien apprendre. D'ou le choix du **F1-Score**.

<p align="center">
  <img src="assets/images/target_distribution.png" width="400"/>
</p>

### Variables clefs

**Pays et source de trafic :**

<p align="center">
  <img src="assets/images/conversion_by_country_source.png" width="800"/>
</p>

- La **Chine** et les **US** affichent des taux de conversion legerement superieurs.
- La **source de trafic** (Ads, SEO, Direct) a un impact marginal sur la conversion.

**Age et engagement :**

<p align="center">
  <img src="assets/images/age_distribution_and_pages.png" width="600"/>
  <img src="assets/images/age_distribution_and_pages_2.png" width="600"/>
</p>

- Les **jeunes (18-30 ans)** convertissent davantage.
- Le **nombre de pages visitees** (`total_pages_visited`) est le facteur le plus discriminant : les utilisateurs qui convertissent visitent en mediane ~15 pages, contre ~4 pour les non-convertis.

### Nettoyage
2 lignes avec un age de 123 ans ont ete supprimees (erreurs de saisie). Le dataset final contient **284 578 lignes**.

## Preprocessing

| Etape | Methode | Detail |
|-------|---------|--------|
| Encodage | `pd.get_dummies(drop_first=True)` | One-Hot Encoding de `country` et `source` |
| Split | `train_test_split(stratify=y)` | 80% train / 20% test, stratifie |
| Normalisation | `StandardScaler` | fit sur le train, transform sur le test |

## Modelisation

### Modeles testes

| # | Modele | Type | Gestion desequilibre |
|---|--------|------|---------------------|
| 1 | **Logistic Regression** | Lineaire (Baseline) | `class_weight='balanced'` |
| 2 | **Random Forest** | Ensemble - Bagging | `class_weight='balanced'` |
| 3 | **XGBoost** (defaut) | Ensemble - Boosting | Parametres par defaut |
| 4 | **XGBoost** (optimise) | Ensemble - Boosting | GridSearchCV |

### Resultats

| Modele | Precision | Recall | F1 (Test) | F1 (CV 3-fold) |
|--------|:---------:|:------:|:---------:|:--------------:|
| Logistic Regression | 0.35 | 0.94 | 0.5118 | 0.5111 +/- 0.0014 |
| Random Forest | 0.44 | 0.83 | 0.5761 | 0.5754 +/- 0.0053 |
| XGBoost (defaut) | 0.85 | 0.68 | 0.7544 | 0.7561 +/- 0.0074 |
| **XGBoost (optimise)** | **0.85** | **0.69** | **0.7591** | **0.7650 +/- 0.0069** |

### Matrices de Confusion

<p align="center">
  <img src="assets/images/confusion_matrix_logreg.png" width="350"/>
  <img src="assets/images/confusion_matrix_rf.png" width="350"/>
</p>
<p align="center">
  <img src="assets/images/confusion_matrix_xgboost.png" width="350"/>
  <img src="assets/images/confusion_matrix_xgboost_optimized.png" width="350"/>
</p>

**Lecture :** La Logistic Regression a un recall eleve (94%) mais genere beaucoup de faux positifs (precision = 35%). Le XGBoost optimise trouve le meilleur equilibre precision/recall.

### Optimisation (GridSearchCV)

Grille testee sur XGBoost avec cross-validation 3-fold :

| Parametre | Valeurs testees | Retenu |
|-----------|----------------|--------|
| `max_depth` | 3, 5, 7 | **7** |
| `learning_rate` | 0.05, 0.1, 0.2 | **0.1** |
| `n_estimators` | 100, 200 | **100** |

18 combinaisons x 3 folds = **54 fits**. Metrique d'optimisation : **F1-Score**.

## Feature Importance

<p align="center">
  <img src="assets/images/feature_importance.png" width="600"/>
</p>

### Leviers d'action identifies

| Variable | Impact | Recommandation |
|----------|--------|---------------|
| `total_pages_visited` | Predicteur dominant | Inciter la navigation : liens internes, suggestions d'articles, contenu interactif |
| `age` | Les 18-30 ans convertissent mieux | Cibler les campagnes marketing sur cette tranche |
| `new_user` | Les recurrents convertissent mieux | Strategie de retargeting pour faire revenir les visiteurs |
| `country` | Disparites geographiques | Adapter le contenu ou le timing par region |

## Installation

```bash
git clone https://github.com/athanormark/Conversion_Rate-BLOC-3_JEDHA_FORMATION.git
cd Conversion_Rate-BLOC-3_JEDHA_FORMATION
pip install -r requirements.txt
```

Placer `conversion_data_train.csv` et `conversion_data_test.csv` dans `data/raw/`, puis :

```bash
jupyter notebook notebooks/1.0-eda-model-training.ipynb
```

## Structure du Projet

```text
conversion_rate_project/
├── data/
│   ├── raw/                  # Donnees brutes (non versionnees)
│   └── processed/            # submission.csv generee
├── notebooks/
│   └── 1.0-eda-model-training.ipynb
├── assets/
│   └── images/               # Graphiques extraits du notebook
├── .gitignore
├── requirements.txt
└── README.md
```

## Auteur
Athanor SAVOUILLAN
