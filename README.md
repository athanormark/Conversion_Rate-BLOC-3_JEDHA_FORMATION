# Conversion Rate Challenge -- Prediction de la conversion newsletter

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=fff)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=fff)](#)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=fff)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=flat)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=flat)](#)
[![JEDHA](https://img.shields.io/badge/JEDHA-blueviolet?style=flat)](#)

---

## About

**datascienceweekly.org** est une newsletter curatee par des data scientists independants. L'equipe souhaite comprendre le comportement des visiteurs de son site et predire si un visiteur va s'abonner a la newsletter a partir de son profil et de sa navigation.

L'objectif est double :
1. **Construire un modele predictif** capable d'identifier les visiteurs les plus susceptibles de convertir, afin d'orienter les actions marketing en consequence.
2. **Analyser les parametres du modele** pour decouvrir des leviers d'action concrets permettant d'ameliorer le taux de conversion.

Le projet s'inscrit dans un **challenge de Machine Learning** (format Kaggle) : construire le modele avec le meilleur **F1-Score**, puis soumettre les predictions sur un jeu de test non labellise.

**Pourquoi le F1-Score ?** Le dataset est tres desequilibre (3.23 % de conversions). L'Accuracy est inadaptee : un modele naif qui predit toujours 0 atteindrait 96.8 % sans rien detecter. Le F1-Score penalise les modeles qui sacrifient la precision ou le recall.

Projet realise dans le cadre du **BLOC 3 -- Machine Learning** de la formation Data Fullstack (JEDHA Bootcamp).

---

## Dataset

Deux fichiers CSV fournis par l'organisateur du challenge :

| Fichier | Lignes | Role |
|---------|--------|------|
| `conversion_data_train.csv` | 284 580 | Entrainement (labellise) |
| `conversion_data_test.csv` | 31 620 | Soumission (non labellise) |

**6 colonnes, aucune valeur manquante.**

| Variable | Type | Description |
|----------|------|-------------|
| `country` | Categorielle (4) | US, UK, China, Germany |
| `age` | Numerique | 17-123 ans |
| `new_user` | Binaire | 0 = recurrent, 1 = nouveau |
| `source` | Categorielle (3) | Ads, Direct, Seo |
| `total_pages_visited` | Numerique | 1-29 pages |
| `converted` | Binaire (target) | 0 = non, 1 = oui |

Taux de conversion : **3.23 %** (9 186 / 284 580). **Nettoyage** : 2 lignes avec age = 123 ans supprimees. Dataset final : **284 578 lignes**.

---

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

---

## Pipeline

### 1. Exploration (EDA)

Analyse de la distribution de la cible, des correlations entre variables et detection des valeurs aberrantes. Observations principales :
- Le nombre de pages visitees (`total_pages_visited`) est le predicteur le plus discriminant
- Les jeunes (18-30 ans) convertissent legerement mieux
- Disparites geographiques moderees entre les pays

### 2. Preprocessing

| Etape | Methode | Justification |
|-------|---------|---------------|
| Encodage | `pd.get_dummies(drop_first=True)` | One-Hot Encoding. `drop_first` evite la multicolinearite |
| Split | `train_test_split(stratify=y, test_size=0.2)` | Preserve le ratio 96.77/3.23 |
| Normalisation | `StandardScaler` | fit sur le train, transform sur le test (pas de data leakage) |

### 3. Modelisation

4 modeles de complexite croissante :

1. **Logistic Regression** (`class_weight='balanced'`) : baseline lineaire
2. **Random Forest** (100 arbres, `class_weight='balanced'`) : ensemble par bagging
3. **XGBoost** (defaut) : ensemble par boosting
4. **XGBoost** (optimise par `GridSearchCV`) : modele retenu

### 4. Optimisation (GridSearchCV)

Recherche sur grille, cross-validation 3-fold, F1-Score comme metrique :

| Parametre | Valeurs testees | Retenu |
|-----------|-----------------|--------|
| `max_depth` | 3, 5, 7 | **7** |
| `learning_rate` | 0.05, 0.1, 0.2 | **0.1** |
| `n_estimators` | 100, 200 | **100** |

### 5. Submission

Le meilleur modele est applique sur le test set (31 620 lignes) avec le meme preprocessing. Le fichier `submission.csv` est genere dans `data/processed/` (colonne unique `converted`).

---

## Resultats

### Comparaison des modeles

| Modele | Precision | Recall | F1 (Test) | F1 (CV 3-fold) |
|--------|:---------:|:------:|:---------:|:--------------:|
| Logistic Regression (baseline) | 0.35 | 0.94 | 0.5118 | 0.5111 +/- 0.0014 |
| Random Forest | 0.44 | 0.83 | 0.5761 | 0.5754 +/- 0.0053 |
| XGBoost (defaut) | 0.85 | 0.68 | 0.7544 | 0.7561 +/- 0.0074 |
| **XGBoost (optimise)** | **0.85** | **0.69** | **0.7591** | **0.7650 +/- 0.0069** |

Progression baseline → meilleur modele : **+48 %** de F1-Score. Scores CV et test proches : pas d'overfitting.

### Feature Importance et leviers d'action

| Variable | Impact | Recommandation |
|----------|--------|---------------|
| `total_pages_visited` | Predicteur dominant. Au-dela de 12-15 pages, conversion quasi-certaine | Inciter la navigation : liens internes, suggestions d'articles |
| `age` | Les 18-30 ans convertissent davantage | Cibler les campagnes marketing sur cette tranche |
| `new_user` | Les utilisateurs recurrents convertissent mieux | Strategie de retargeting pour faire revenir les visiteurs |
| `country` | Disparites geographiques moderees | Adapter le contenu ou le timing par region |

---

## Conclusion

Le projet repond a la double problematique : **predire quels visiteurs vont s'abonner**, et **identifier les leviers d'action** pour ameliorer le taux de conversion.

- Le **XGBoost optimise** atteint un F1 de **0.759**, partant d'un baseline a 0.512 (+48 %)
- Le nombre de **pages visitees** est le predicteur dominant
- Les **18-30 ans** et les **utilisateurs recurrents** convertissent davantage

**Leviers d'action pour l'equipe** :
- Inciter la navigation (liens internes, contenu interactif) pour augmenter le nombre de pages vues
- Cibler les campagnes sur les 18-30 ans
- Mettre en place du retargeting pour faire revenir les visiteurs
- Adapter le contenu par region geographique

---

## Structure du projet

```text
conversion_rate_project/
├── data/
│   ├── raw/                  # Donnees brutes (non versionnees)
│   └── processed/            # submission.csv
├── notebooks/
│   └── 1.0-eda-model-training.ipynb
├── assets/
│   └── images/               # Graphiques du notebook
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Auteur

Athanor SAVOUILLAN · [GitHub](https://github.com/athanormark)
