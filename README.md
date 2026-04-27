# Generative Quant Vision System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-00599C.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-LibTorch-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

> **Une nouvelle perspective sur les marchés financiers : transformer les séries temporelles (1D) en matrices spatiales (2D) pour extraire, interpréter et simuler les dynamiques boursières grâce à la Vision Artificielle.**

## Présentation du Projet

Les modèles quantitatifs traditionnels (1D) sont souvent vulnérables au micro-bruit des marchés financiers. Ce projet propose une approche basée sur la **Vision par Ordinateur**. En encodant les fenêtres de prix du S&P 500 sous forme d'images **GAF (Gramian Angular Field)**, les variations de marché deviennent des textures géométriques complexes. Les réseaux de neurones convolutifs (CNN) peuvent alors identifier des modèles structurels (patterns) invisibles à l'œil nu.

### Fonctionnalités Clés
1. **Encodage Spatial 1D $\rightarrow$ 2D :** Transformation des cours de clôture en champs angulaires gramiens (GAF).
2. **Détection d'Anomalies (Auto-encodeur) :** Évaluation de la normalité structurelle du marché via le score SSIM (Structural Similarity Index).
3. **Prédiction de Tendance (CNN) :** Classification binaire pour prédire la direction du marché à J+1.
4. **Interprétabilité (Grad-CAM) :** Ouverture de la "boîte noire" de l'IA en générant des cartes de chaleur soulignant les corrélations temporelles ayant déclenché la décision.
5. **Génération de Scénarios (VAE) :** Remplacement des simulations de Monte-Carlo classiques par un Auto-encodeur Variationnel capable de générer des "univers parallèles" réalistes basés sur la signature latente du marché actuel.
6. **Moteur d'Inférence Basse Latence (C++) :** Déploiement industriel du modèle prédictif en C++ pur via LibTorch, garantissant une exécution déterministe pour le Trading Haute Fréquence (HFT).

---

## Architecture du Pipeline

* `dataset_builder.py` : Ingestion des données via Yahoo Finance et transformation GAF.
* `autoencoder_model.py` / `vae_model.py` : Architectures neuronales (PyTorch).
* `train.py` / `train_classifier.py` / `train_vae.py` : Scripts d'entraînement séquentiels et exportation `TorchScript`.
* `gradcam_visualizer.py` : Outil d'extraction des gradients spatiaux.
* `app.py` : Dashboard interactif complet propulsé par Streamlit.
* `main.cpp` : Moteur d'inférence en C++ pur.

---

## Aperçu Visuel

### 1. Interprétabilité du modèle (Grad-CAM)
L'IA met en évidence (en rouge) les interactions temporelles clés qui motivent sa prédiction, prouvant qu'elle identifie de réelles structures de corrélation et non du bruit aléatoire.
*(Voir `Images/gradcam_analysis.png`)*

### 2. Reconstruction et Détection d'Anomalies
Comparaison entre la matrice de marché originale et la reconstruction débruitée par l'Auto-encodeur. Un score SSIM faible indique un régime de marché inédit ou une anomalie (krach).
*(Voir `Images/reconstruction_comparison.png`)*

### 3. Marchés Synthétiques (VAE)
Simulation de scénarios de stress-tests : génération de variations réalistes d'une période boursière spécifique par échantillonnage dans l'espace latent $\mathcal{N}(\mu, \sigma^2)$.
*(Voir `Images/synthetic_markets_vae.png`)*

---

## Installation et Déploiement

### 1. Environnement Python (Recherche & Entraînement)
```bash
# Clonage et installation des dépendances
git clone [https://github.com/arnbskr/generative-quant-vision.git](https://github.com/arnbskr/generative-quant-vision.git)
cd generative-quant-vision
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Lancement du Dashboard (Interface Interactive)

Pour explorer le pipeline de bout en bout visuellement :

```bash
streamlit run app.py
```

### 3. Compilation du Moteur C++ (Production Basse Latence)

Prérequis : Télécharger l'API C++ libtorch depuis le site officiel de PyTorch et l'extraire à la racine du projet.

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..
make
./quant_engine
```

## Benchmark de Performance (Latence)

Le passage de l'environnement de recherche (Python) à l'environnement de production (C++) permet de s'affranchir du Global Interpreter Lock (GIL) et de la gestion dynamique de la mémoire (Garbage Collector), éliminant ainsi le risque de Jitter (pics de latence).
- Latence moyenne d'inférence (C++ LibTorch) : `~4.68 ms` (4 681 µs) après warm-up JIT.
- Format du modèle : `TorchScript` optimisé.
- Avantage : Latence strictement déterministe, indispensable pour l'intégration dans une architecture de trading haute fréquence (HFT).