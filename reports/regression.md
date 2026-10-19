# Non-régression : ancien pickle vs nouveaux artefacts

Généré par `python scripts/compare_legacy.py`.

## Taille et chargement

| | Ancien (`recommender_data3.pkl`) | Nouveau (`artifacts/`) |
| :-- | --: | --: |
| Taille | 618.1 Mo | 12.2 Mo |
| Temps de chargement | 0.2 s | 0.11 s |
| RAM maximale du processus | 708 Mo | 180 Mo |

## Accord par signal (300 films, lignes de similarité)

| Signal | Corrélation | Recouvrement top-20 |
| :-- | --: | --: |
| `sbert` | 0.876 | 64% |
| `keywords` | 0.621 | 34% |
| `vibe` | 0.499 | 19% |
| `genre` | 0.853 | 26% |
| `credits` | 0.951 | 61% |
| `origin` | 0.912 | 20% |
| `context` | 0.765 | 19% |
| `collab` | 1.000 | 99% |

## Top-10 du moteur complet (50 films)

Jaccard moyen : **0.29** (médiane 0.29).

Les écarts s'expliquent par :

- `collab` est reconstruit à l'identique (cosinus item-item sur les notes brutes MovieLens) ;
- `vibe` n'était pas versionné : il est reconstruit (accroche + genres + mots-clés, MiniLM), c'est la principale source d'écart ;
- les métadonnées viennent de l'API TMDB actuelle (synopsis, mots-clés et popularité ont changé depuis l'export Kaggle de 2017) ;
- les règles « langue étrangère » et « film populaire » sont maintenant actives ;
- le cold start redistribue le poids réel de `collab` (et non une constante de 0,2) ;
- 8 films supprimés de TMDB sont sortis du catalogue (6 195 films).
