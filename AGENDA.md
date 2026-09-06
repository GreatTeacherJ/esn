# Infrastructure du Projet ESN

**Modèle de Prédiction de Démission d'Employés**

---

## Vue d'ensemble

Ce projet est un système de machine learning conçu pour prédire les risques de démission d'employés dans une entreprise. Il utilise Python 3.13+ avec un écosystème data science complet.

## Objectif Principal

Entraîner un modèle de machine learning pour prédire si des employés vont démissionner d'une entreprise, permettant une gestion préventive des ressources humaines.

---

## Structure du Projet

```
esn/
├── .git/                          # Contrôle de version (Git)
├── .venv/                         # Environnement virtuel Python
├── .gitignore                     # Fichiers à ignorer
├── .python-version                # Version Python fixée (3.13)
├── pyproject.toml                 # Configuration du projet et dépendances
├── uv.lock                        # Lock file des dépendances (uv package manager)
│
├── README.md                      # Documentation projet (actuellement vide)
├── AGENDA.md                      # Plan de travail par phases
├── INFRASTRUCTURE.md              # Ce fichier (description infrastructure)
│
├── src/                           # Code source principal
│   ├── esn/                       # Package principal
│   │   ├── __init__.py            # Point d'entrée du package
│   │   └── __pycache__/           # Cache Python compilé
│   │
│   └── app/                       # Application et notebooks
│       ├── data.ipynb             # Notebook Jupyter - Exploration/Prétraitement des données
│       │
│       └── data/                  # Répertoire de données
│           ├── extrait_eval.csv   # Données d'évaluation des employés
│           ├── extrait_sirh.csv   # Données SIRH (Système d'Info Ressources Humaines)
│           └── extrait_sondage.csv # Données de sondage employés
│
└── rapport IA/                    # Rapports d'analyse et résultats
    ├── rapport_code_ml_rh.md      # Rapport principal du modèle ML et RH
    └── rapport_shap_best_estimator.md # Rapport d'interprétabilité SHAP du meilleur modèle
```

---

## Configuration Technique

### Langage et Environnement

- **Langage** : Python 3.13+
- **Gestionnaire de dépendances** : `uv` (UV build)
- **Gestionnaire d'environnement virtuel** : `.venv`
- **Version Python fixée** : `.python-version` (3.13)

### Dépendances Principales

#### Data Science et Machine Learning

- **pandas** (≥3.0.5) - Manipulation et analyse des données
- **numpy** (≥2.5.2) - Calculs numériques
- **scikit-learn** (≥1.9.0) - Algorithmes ML et preprocessing

#### Visualisation

- **matplotlib** (≥3.11.1) - Graphiques statiques
- **plotly[express]** (≥7.0.0) - Graphiques interactifs

#### Notebook et Développement

- **nbformat** (≥5.11.1) - Format des notebooks Jupyter
- **ipykernel** (≥7.3.0) - Kernel Jupyter (groupe dev)

### Point d'Entrée

- Commande CLI : `esn` (définie dans `pyproject.toml`)
- Fonction d'entrée : `esn.main()` (dans `src/esn/__init__.py`)

---

## Sources de Données

Trois fichiers CSV contenant les données brutes :

### 1. `extrait_eval.csv`

**Source** : Système d'évaluation des performances

- **Contient** : Évaluations de performance, notes, objectifs
- **Granularité** : Données par employé et période
- **Lien avec démission** : Performance et satisfaction au travail

### 2. `extrait_sirh.csv`

**Source** : Système d'Information RH

- **Contient** : Données administratives, salaire, poste, ancienneté, contrat
- **Granularité** : Informations actuelles et historiques par employé
- **Lien avec démission** : Conditions d'emploi, mobilité interne

### 3. `extrait_sondage.csv`

**Source** : Sondages internes d'engagement

- **Contient** : Satisfaction, bien-être, motivation, environnement de travail
- **Granularité** : Réponses à des questions par employé
- **Lien avec démission** : Indicateurs directs de satisfaction

---

## Dossiers Clés

### `/src/esn/`

**Package Python principal**

- Point d'entrée : `__init__.py`
- Contient la fonction `main()`
- À développer : modules de modélisation, utility functions

### `/src/app/`

**Applications et notebooks**

- `data.ipynb` : Exploration interactive des données
- `data/` : Répertoire des fichiers de données source

### `/rapport IA/`

**Documentation des résultats**

- Rapports Markdown avec analyses, métriques, visualisations
- Explications du modèle pour les stakeholders

---

## Outils et Technologies

| Catégorie           | Outils                          |
| ------------------- | ------------------------------- |
| **Control Version** | Git (.git/)                     |
| **Package Manager** | UV                              |
| **Environment**     | Python venv                     |
| **Notebooks**       | Jupyter (IPython)               |
| **Data Processing** | Pandas, NumPy                   |
| **ML & Modeling**   | Scikit-learn, XGBoost, LightGBM |
| **Visualization**   | Matplotlib, Plotly              |
| **Explainability**  | SHAP                            |

---

## Dépendances du Projet

### Production (`dependencies`)

```
matplotlib>=3.11.1      # Visualisations
nbformat>=5.11.1        # Support Jupyter
numpy>=2.5.2            # Calculs numériques
pandas>=3.0.5           # DataFrames et manipulation
plotly[express]>=7.0.0  # Graphiques interactifs
scikit-learn>=1.9.0     # Modèles ML
```

### Développement (`dev` group)

```
ipykernel>=7.3.0        # Kernel Jupyter pour exécution interactive
```

### Plate-forme

- OS : Windows (chemins avec backslash)
- Python Version : 3.13
- Date de création : 2026
