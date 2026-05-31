<<<<<<< HEAD
# scanres
=======
# Network Security Analyzer - SVD/PCA

Application complète d'analyse de réseau par Décomposition en Valeurs Singulières (SVD) et Analyse en Composantes Principales (PCA) pour la détection de vulnérabilités, la modélisation de perturbations stochastiques et la simulation d'évasion IDS.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN APPLICATION                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Basic Mode  │  │Offensive Mode│  │  Full Mode   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
    ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
    │  Unified  │    │ Proactive │    │  IDS      │
    │  SVD      │    │ Intelligence│   │ Evasion   │
    │  Engine   │    │  Engine   │    │ Simulator │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Visualizer    │
                    │  (Matplotlib)   │
                    └────────────────┘
```

## Fichiers

| Fichier | Description |
|---------|-------------|
| `main_application.py` | Point d'entrée avec CLI |
| `network_svd_analyzer.py` | Analyse SVD/PCA de base |
| `optimized_offensive_engine.py` | Moteur offensif optimisé |
| `requirements.txt` | Dépendances Python |

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Mode Analyse de Base
Analyse SVD/PCA standard avec détection de vulnérabilités et reconstruction de réseau.

```bash
python main_application.py --mode basic --nodes 50
```

### Mode Intelligence Offensive
Scan proactif des vulnérabilités, évasion IDS et attaques par transfer learning.

```bash
python main_application.py --mode offensive --nodes 150
```

### Mode Complet
Exécute les deux modes séquentiellement.

```bash
python main_application.py --mode full --nodes 150 --seed 42
```

### Options

| Option | Description | Défaut |
|--------|-------------|--------|
| `--mode` | Mode d'exécution | `full` |
| `--nodes` | Nombre de nœuds | `150` |
| `--seed` | Seed aléatoire | `42` |
| `--output` | Répertoire de sortie | `./output` |

## Fonctionnalités

### Analyse SVD/PCA
- Décomposition SVD dense et sparse
- Cache intelligent des décompositions
- Mise à jour incrémentale (algorithme de Brand)
- Extraction de caractéristiques spectrales

### Intelligence Proactive
- Score composite de vulnérabilité (6 métriques)
- Détection d'anomalies (Isolation Forest + LOF)
- Prédiction du temps de compromission
- Génération de perturbations optimales

### Évasion IDS
- 3 stratégies adaptatives (direct, null_space, temporal)
- Apprentissage adversarial des seuils IDS
- Budget d'attaque optimisé (ε = 0.25)

### Transfer Learning
- Matrice de rotation Procrustes
- Adaptation spectrale nœud→nœud
- Évaluation de l'efficacité de transfert

## Structure du Réseau

Le réseau généré utilise un modèle hybride :
- **Communautés** : Watts-Strogatz (petits mondes)
- **Ponts** : Connexions inter-communautaires
- **Super-hubs** : Nœuds critiques avec connexions préférentielles
- **Pondération** : Basée sur la centralité d'intermédiarité

## Performance

| Fonction | Temps Moyen | Temps Max |
|----------|-------------|-----------|
| SVD (150×150) | 0.74 ms | 178 ms |
| Scan vulnérabilités | 356 ms | 356 ms |
| Évasion (100 épisodes) | 10 ms | 10 ms |
| Transfer learning | 6 ms | 6 ms |

## Dépendances

- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- networkx >= 2.6.0

## Licence

Ce projet est fourni à des fins éducatives et de recherche.


Voici les commandes de lancement pour exécuter l'application selon les différents modes. 

Assurez-vous d'avoir installé les dépendances nécessaires avant de lancer le programme :
```bash
pip install numpy scipy matplotlib seaborn scikit-learn networkx joblib
```

### 1. Mode Basic (Analyse standard)
Analyse SVD/PCA de base, détection de vulnérabilités et reconstruction du réseau (utilise le module `network_svd_analyzer.py`).
```bash
python main_application.py --mode basic
```
*Avec paramètres personnalisés (ex: 100 nœuds) :*
```bash
python main_application.py --mode basic --nodes 100 --verbose
```

### 2. Mode Offensive (Intelligence proactive)
Scan avancé, détection non-linéaire, évasion IDS par RL (PPO) et transfer learning (utilise le module `optimized_offensive_engine.py`).
```bash
python main_application.py --mode offensive
```
*Pour une simulation plus longue (500 épisodes d'évasion) sur un plus grand réseau :*
```bash
python main_application.py --mode offensive --nodes 500 --evasion-episodes 500
```

### 3. Mode Full (Complet)
Enchaîne le mode Basic puis le mode Offensive de manière séquentielle.
```bash
python main_application.py --mode full
```

### 4. Mode Benchmark (Tests de performance)
Évalue les temps de calcul (initialisation, SVD à froid, SVD avec cache) pour différentes tailles de réseau.
```bash
python main_application.py --mode benchmark
```

---

### Options supplémentaires (utilisables avec n'importe quel mode)

**Désactiver le profiling et rendre l'exécution silencieuse :**
```bash
python main_application.py --mode full --no-profile --no-verbose
```

**Spécifier un répertoire de sortie personnalisé pour les dashboards (images) :**
```bash
python main_application.py --mode offensive --output ./mes_resultats
```

**Forcer l'utilisation des matrices sparse (recommandé si > 100 nœuds) et augmenter le budget d'attaque :**
```bash
python main_application.py --mode offensive --nodes 1000 --sparse --attack-budget 0.4
```

**Activer la parallélisation Joblib (pour le scan de vulnérabilités) :**
```bash
python main_application.py --mode offensive --nodes 500 --parallel
```

### Aide intégrée
Pour voir la liste complète des arguments disponibles directement dans le terminal :
```bash
python main_application.py --help
```
>>>>>>> bf6e2fa (Premier commit)
