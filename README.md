# Prédiction du prix des montres de luxe — ML & Deep Learning

Mini-projet du cours *Machine Learning & Deep Learning* — Eugenia School, MSc 2025-2026 — Ronny Douglas, Victorien Alleg.

**Objectif :** estimer le prix d'une montre de luxe (régression) et la classer *Accessible* (< 8 400 $) ou *Premium* (≥ 8 400 $).

| Notebook | Contenu |
| --- | --- |
| `01_ML_prix_montres.ipynb` | Nettoyage, analyse exploratoire, régression simple / multiple, Random Forest, validation croisée, régression logistique, prédictions |
| `02_DL_prix_montres.ipynb` | Réseau de neurones PyTorch (MLP) pour le prix et la gamme, comparaison avec le ML classique |

## Résultats (jeu de test, 102 montres)

| Modèle | R² prix | RMSE | Accuracy gamme |
| --- | ---: | ---: | ---: |
| Régression linéaire simple (diamètre) | 0,02 | 10 099 $ | — |
| Régression linéaire multiple | 0,78 | 4 786 $ | — |
| **Random Forest** | **0,88** | **3 582 $** | — |
| Régression logistique | — | — | **0,92** |
| MLP PyTorch | 0,87 | 3 715 $ | 0,91 |

Validation croisée 5 plis du Random Forest : R² = 0,82 ± 0,04.

## Historique des versions

- **Octobre 2025** : analyse initiale du mini-projet (rapport HTML, joint en annexe du dossier de certification).
- **Novembre 2026** : notebooks ré-exécutés de bout en bout pour le dossier de certification RNCP, avec une graine aléatoire fixée (42), une validation croisée et un notebook Deep Learning en plus.

Le découpage train / test et quelques réglages diffèrent légèrement de la version d'octobre 2025, d'où de petits écarts de chiffres. Les conclusions sont inchangées.

| Indicateur | Oct. 2025 | Nov. 2026 |
| --- | ---: | ---: |
| R² Random Forest | 88 % | 87,6 % |
| R² régression multiple | 51 % | 77,9 % |
| Précision de la régression logistique | 95 % | 92,2 % |

Ce sont les chiffres de novembre 2026 qui font foi.

## Reproduire

```bash
python -m venv .venv && .venv/bin/pip install pandas numpy scipy scikit-learn matplotlib seaborn torch jupyter
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_ML_prix_montres.ipynb 02_DL_prix_montres.ipynb
```

`build_notebooks.py` régénère les notebooks depuis leur source.

## Données

*Luxury Watch Prices* (Kaggle) — `data/luxury_watch.csv`, 507 montres × 14 variables. Jeu public repris à des fins pédagogiques.
