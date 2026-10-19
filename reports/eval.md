# CineMatch : évaluation hors-ligne

Généré par `python -m evaluation.run` le 2026-10-19T20:47:55+00:00 (graine 42). Résultats bruts : [`eval.json`](eval.json).

## Protocole

- **dataset** : MovieLens ml-latest-small, restreint au catalogue
- **split** : temporel par utilisateur, 80 % train / 20 % test
- **positive** : note >= 4
- **query** : jusqu'à 5 films aimés les plus récents du train par utilisateur
- **relevant** : films aimés dans la partie test
- **excluded from ranking** : film requête + films déjà notés dans le train
- **collab** : recalculé sur la partie train uniquement
- **users split** : 20% validation (poids), reste test (une seule passe)
- **cold start** : simulé : signal collaboratif du film requête masqué, poids redistribué
- **bootstrap** : 1000 tirages sur les utilisateurs, IC 95 %
- **n val queries** : 581
- **n test queries** : 2330
- **n test users** : 472
- **catalog size** : 6195
- **catalog cold share** : 0.4728

Le *cold start simulé* rejoue chaque requête en masquant le signal collaboratif du film requête, comme pour les 43 % du catalogue sans note MovieLens.

## Baselines vs hybride (standard)

Moyenne [IC 95 % bootstrap].

| Méthode | NDCG@10 | HitRate@10 | MRR@10 | P@4 | R@10 | Couverture | Diversité | Popularité (centile) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Popularité | 0.087 [0.074–0.101] | 0.354 [0.314–0.397] | 0.181 | 0.084 | 0.066 | 1.4% | 0.83 | 98 |
| Aléatoire | 0.002 [0.001–0.002] | 0.014 [0.009–0.019] | 0.004 | 0.002 | 0.001 | 97.5% | 0.85 | 49 |
| SBERT seul | 0.010 [0.008–0.013] | 0.051 [0.043–0.061] | 0.027 | 0.009 | 0.007 | 65.9% | 0.65 | 55 |
| Collaboratif seul | 0.082 [0.074–0.091] | 0.352 [0.325–0.378] | 0.153 | 0.065 | 0.082 | 40.5% | 0.82 | 81 |
| Hybride, poids uniformes | 0.042 [0.037–0.048] | 0.224 [0.205–0.245] | 0.088 | 0.034 | 0.041 | 49.3% | 0.76 | 74 |
| Hybride, poids de base sans règles | 0.044 [0.039–0.050] | 0.223 [0.203–0.244] | 0.092 | 0.036 | 0.042 | 50.0% | 0.75 | 75 |
| Hybride actuel (base + règles) | 0.057 [0.051–0.064] | 0.276 [0.254–0.298] | 0.114 | 0.046 | 0.055 | 48.2% | 0.76 | 77 |
| Hybride optimisé (Optuna) | 0.070 [0.062–0.076] | 0.318 [0.290–0.341] | 0.135 | 0.059 | 0.067 | 42.4% | 0.82 | 79 |

## Baselines vs hybride (cold start simulé)

Moyenne [IC 95 % bootstrap].

| Méthode | NDCG@10 | HitRate@10 | MRR@10 | P@4 | R@10 | Couverture | Diversité | Popularité (centile) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Popularité | 0.087 [0.074–0.101] | 0.354 [0.314–0.397] | 0.181 | 0.084 | 0.066 | 1.4% | 0.83 | 98 |
| Aléatoire | 0.002 [0.001–0.002] | 0.014 [0.009–0.019] | 0.004 | 0.002 | 0.001 | 97.5% | 0.85 | 49 |
| SBERT seul | 0.010 [0.008–0.013] | 0.051 [0.043–0.061] | 0.027 | 0.009 | 0.007 | 65.9% | 0.65 | 55 |
| Collaboratif seul | 0.004 [0.002–0.006] | 0.034 [0.017–0.051] | 0.008 | 0.004 | 0.003 | 4.6% | 0.85 | 54 |
| Hybride, poids uniformes | 0.024 [0.020–0.027] | 0.128 [0.112–0.145] | 0.056 | 0.020 | 0.020 | 56.0% | 0.74 | 67 |
| Hybride, poids de base sans règles | 0.016 [0.013–0.019] | 0.083 [0.071–0.097] | 0.039 | 0.014 | 0.013 | 61.0% | 0.72 | 62 |
| Hybride actuel (base + règles) | 0.017 [0.014–0.021] | 0.091 [0.077–0.105] | 0.041 | 0.015 | 0.014 | 61.1% | 0.72 | 62 |
| Hybride optimisé (Optuna) | 0.036 [0.032–0.041] | 0.191 [0.172–0.211] | 0.076 | 0.028 | 0.035 | 50.6% | 0.79 | 72 |

## Optimisation des poids (Optuna)

200 essais TPE sur la validation, objectif : moyenne NDCG@10 standard et cold start (validation). Validation : 0.0361 (poids manuels) → 0.0554 (poids optimisés).

| Signal | Poids manuel | Poids optimisé |
| :-- | --: | --: |
| Histoire (sbert) | 0.15 | 0.03 |
| Sujets (keywords) | 0.25 | 0.17 |
| Ambiance (vibe) | 0.15 | 0.02 |
| Genre | 0.10 | 0.01 |
| Casting (credits) | 0.10 | 0.04 |
| Origine | 0.00 | 0.06 |
| Époque & popularité (context) | 0.05 | 0.39 |
| Public (collab) | 0.20 | 0.28 |

## Ablation (test)

Variation de NDCG@10, en points, quand on retire un signal ou une règle de l'hybride actuel. Négatif : l'élément aide.

![Ablation](ablation.png)

| Élément retiré | Standard | Cold start simulé |
| :-- | --: | --: |
| Histoire (sbert) | +0.31 | +0.11 |
| Sujets (keywords) | +0.32 | +0.15 |
| Ambiance (vibe) | +0.36 | +0.14 |
| Genre | +0.39 | +0.11 |
| Casting (credits) | +0.01 | -0.12 |
| Origine | +0.04 | +0.01 |
| Époque & popularité (context) | -0.47 | -0.31 |
| Public (collab) | -3.74 | +0.00 |
| Règle « franchise » | -0.01 | +0.02 |
| Règle « horror » | +0.04 | +0.00 |
| Règle « foreign » | +0.04 | +0.02 |
| Règle « popular » | -1.27 | -0.14 |

## Lecture

- *Couverture* : part du catalogue apparaissant dans au moins un top-10.
- *Diversité* : 1 − similarité SBERT moyenne entre les films d'une même liste.
- *Popularité* : centile moyen (nombre de votes TMDB) des films recommandés ; 50 = catalogue moyen.
