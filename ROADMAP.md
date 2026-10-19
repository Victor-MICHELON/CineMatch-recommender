# CineMatch — Roadmap d'amélioration

État de départ : commit `1d008ad` (« rebuild project »), octobre 2026.

Objectif : passer d'une démo fonctionnelle à un projet **reproductible, évalué et défendable en entretien**, avec des chiffres mesurés à mettre sur le CV.

Les phases sont ordonnées par rapport valeur / effort. Les phases 0 à 2 sont le minimum à faire. Les phases 3 à 5 rendent le projet remarquable.

| Phase | Thème | Effort estimé | Débloque |
| :-- | :-- | :-- | :-- |
| 0 | Correctifs immédiats | ½ jour | Démo en ligne stable |
| 1 | Pipeline de données reproductible | 2–3 jours | Règles inactives, artefacts légers |
| 2 | Évaluation hors-ligne | 2–3 jours | **KPIs du CV** |
| 3 | Qualité du code & CI | 1–2 jours | Crédibilité « ingénierie » |
| 4 | Fonctionnalités produit & UI | 2–4 jours | Démo plus convaincante |
| 5 | Passage à l'échelle (optionnel) | 3–5 jours | Argument « production » |
| 6 | Valorisation (README, CV, portfolio) | ½ jour | Visibilité |

---

## État d'avancement (19 octobre 2026)

Phases 0 à 4 réalisées, phase 6 en partie (README). Phase 5 non commencée (optionnelle).

| Point | Résultat mesuré | Source |
| :-- | :-- | :-- |
| Artefacts | 618,1 Mo → **12,2 Mo** (−98 %), RAM du processus 708 → 180 Mo | `reports/regression.md` |
| Latence `recommend()` | p50 1,4 ms, p95 2,2 ms (chargement 0,12 s) | `reports/bench.json` |
| NDCG@10 standard | hybride manuel 0,057, optimisé 0,070 ; popularité 0,087 ; collaboratif seul 0,082 | `reports/eval.json` |
| Cold start (HitRate@10) | hybride optimisé 0,191 vs collaboratif seul 0,034 (×5,6) | `reports/eval.json` |
| Tests | 67 tests, couverture 99 % sur `cinematch/`, ruff + mypy strict | `pytest --cov` |

**Écarts assumés par rapport au plan :**
- **0.1** : les logs Streamlit Cloud n'ont pas été consultés (accès au compte requis). Cause probable mesurée en local : téléchargement LFS de 618 Mo et ~700 Mo de RAM pour une limite d'environ 1 Go.
- **1.2.4** : plutôt que de tronquer au top-200 voisins, on stocke les **facteurs** (matrices creuses de caractéristiques normalisées). `F @ F[i]` redonne le cosinus exact, sans perte et pour moins de place.
- **1.2.2** : les CSV Kaggle sont remplacés par l'API TMDB (`pipeline/01_fetch_tmdb.py`). 8 des 6 203 films n'existent plus sur TMDB, il en reste donc 6 195.
- **`vibe`** : le code d'origine est introuvable. Le signal est reconstruit à partir de l'accroche, des genres et des mots-clés (corrélation de 0,50 avec l'ancien signal). `collab` est retrouvé à l'identique (corrélation de 1,000).
- **2.1** : 20 % des utilisateurs en validation (au lieu de 10 %), car `ml-latest-small` n'en compte que 610. Jusqu'à 5 requêtes par utilisateur.
- **1.2.8** : `artifacts.zip` est produit par `python -m pipeline.build --zip`, mais **la GitHub Release `artifacts-v1` reste à publier** (l'app la télécharge au premier lancement).
- **6.1** : le GIF de démo et de nouvelles captures d'écran restent à faire. **6.2, 6.3** : à faire.

---

## Constat (mesuré sur le repo actuel)

| Point | Observation | Impact |
| :-- | :-- | :-- |
| Artefact de données | `recommender_data3.pkl` = 618 Mo (8 matrices denses 6 203² en float16), via Git LFS | Déploiement fragile (quota LFS, mémoire), `pickle` = exécution de code arbitraire au chargement |
| Matrices très creuses | `credits` 99,6 % de zéros, `keywords` 96,2 %, `collab` 82,6 %, `genre` 52,1 % | Stockage dense inutile : un format creux diviserait la taille par > 10 |
| Règles inactives | Le DataFrame n'a que `id, title, year, director, genres_list, overview` : pas de `original_language` ni `vote_count` | Les règles « langue étrangère » et « film populaire » de `smart_weights` ne s'appliquent jamais |
| Pipeline incomplet | Les signaux `vibe` et `collab`, la fusion des ~1 400 films récents et l'export du pickle ne sont pas versionnés | Projet non reproductible |
| Notebooks | 3 notebooks qui se recoupent (`hybrid-dual-tower-…`, `hybrid-movie-reco-…`, `hybrid-weighted-…`), dont 2 sans description | Confusion pour un lecteur |
| `recent_movies.csv` | 52 Mo commités en Git classique (hors LFS) | Proche de la limite GitHub de 100 Mo, alourdit chaque clone |
| Évaluation | Aucune métrique de qualité | Impossible de chiffrer le projet |
| `app.py` | 512 lignes : CSS, chargement, appels TMDB et UI mélangés | Difficile à maintenir et à tester |
| Dev Container | Python 3.11, installe `streamlit` en double, désactive CORS/XSRF | Environnement différent du local (3.14) et du déploiement |
| README | `screenshot1_app.png` n'est plus référencé | Mineur |
| Secrets | Clé TMDB réelle dans `.streamlit/secrets.toml` (ignoré, **jamais commité**, vérifié dans l'historique) | OK, rien à faire |

---

## Phase 0 — Correctifs immédiats (½ jour)

**But :** une démo en ligne qui se charge.

- [ ] **0.1 Diagnostiquer l'app en ligne.** Ouvrir *Manage app → Logs* sur Streamlit Cloud et identifier la cause : pointeur LFS non résolu, `MemoryError` / redémarrage pour mémoire, ou incompatibilité de version de pandas.
- [x] **0.2 Figer les versions.** Créer un `requirements.lock` (ou épingler les versions exactes) avec les versions testées en local : `pandas==3.0.5`, `numpy==2.5.3`, etc. Ajouter un fichier `.python-version` pour que Streamlit Cloud utilise la même version de Python.
- [x] **0.3 Solution de contournement si c'est LFS ou la mémoire.** Héberger l'artefact hors Git (GitHub Release ou Hugging Face Hub) et le télécharger au premier lancement, avec un cache. La phase 1 règle le problème à la racine.
- [x] **0.4 README.** Remettre `screenshot1_app.png` ou supprimer le fichier. Ajouter un badge « statut de la démo ».
- [x] **0.5 Dev Container.** Passer à Python 3.12+, retirer `pip3 install --user streamlit` (déjà dans `requirements.txt`), retirer `--server.enableCORS false --server.enableXsrfProtection false`.

**Critère de fin :** l'URL publique affiche des recommandations pour 3 films testés.

---

## Phase 1 — Pipeline de données reproductible (2–3 jours)

**But :** une seule commande reconstruit toutes les données depuis les sources brutes, en produisant des artefacts légers et sûrs.

### 1.1 Structure cible

```
cinematch/
├── __init__.py
├── config.py           # chemins, hyperparamètres, poids (dataclasses)
├── engine.py           # ex-recommender.py
├── data.py             # chargement des artefacts
└── tmdb.py             # client TMDB (affiches)
pipeline/
├── 01_fetch_tmdb.py    # métadonnées TMDB (catalogue + films récents)
├── 02_clean_merge.py   # nettoyage, fusion, dédoublonnage → catalog.parquet
├── 03_embeddings.py    # SBERT (plot) + signal vibe → embeddings .npy
├── 04_content_sims.py  # keywords, genre, credits, origin, context (creux)
├── 05_collab.py        # MovieLens → similarité item-item (top-N creux)
└── build.py            # orchestre 01→05, écrit artifacts/ + manifest.json
artifacts/              # généré, hors Git (ou publié en Release)
```

Implémenter `pipeline/` sous forme de scripts Python, pas de notebooks. Garder **un seul** notebook d'exploration, et archiver ou supprimer les deux autres après avoir récupéré le code utile.

### 1.2 Tâches

- [x] **1.2.1 Inventorier les 3 notebooks.** Repérer quel code construit quel signal. Documenter `vibe` et `collab` (aujourd'hui « non versionnés ») : modèle, texte d'entrée, paramètres.
- [x] **1.2.2 Catalogue au format Parquet.** `catalog.parquet` avec **toutes** les colonnes utiles, dont `original_language`, `vote_count`, `popularity`, `vote_average`, `runtime`, `tmdb_id` et `movielens_id` (via `links.csv`). Les règles inactives de `smart_weights` redeviennent actives sans changer le code.
- [x] **1.2.3 Embeddings plutôt que matrices denses.** Stocker les signaux sémantiques (`sbert`, `vibe`) sous forme d'**embeddings normalisés** (6 203 × 384, float16 ≈ 4,8 Mo chacun). La similarité d'une requête se calcule par un simple produit `E @ E[i]`, en moins d'une milliseconde.
- [x] **1.2.4 Matrices creuses.** Stocker `keywords`, `credits`, `genre`, `origin` et `collab` en `scipy.sparse` (`.npz`). Si besoin, garder seulement le **top-N voisins** par film (N = 200) pour borner la taille.
- [x] **1.2.5 Signal `context`.** C'est une fonction analytique de l'année et de la popularité. Le calculer à la volée à partir du catalogue au lieu de stocker une matrice 6 203².
- [x] **1.2.6 Abandonner pickle.** Utiliser uniquement Parquet, `.npy` et `.npz`. Ajouter un `manifest.json` (versions, hash SHA-256 de chaque fichier, date, nombre de films) vérifié au chargement.
- [x] **1.2.7 Adapter le moteur.** `recommend()` reçoit un objet `Artifacts` (embeddings, matrices creuses, catalogue) au lieu d'un dict de matrices denses. L'API publique ne change pas, et les tests existants doivent passer.
- [x] **1.2.8 Sortir les données de Git.** Retirer `recommender_data3.pkl` et `recent_movies.csv` du suivi Git. Publier `artifacts/` en GitHub Release ou sur Hugging Face Hub. Ajouter `make data` (ou `python -m pipeline.build`) et un téléchargement automatique au premier lancement de l'app.
- [x] **1.2.9 Vérification de non-régression.** Pour 50 films tirés au hasard, comparer le top-10 de l'ancien pickle avec celui du nouveau pipeline. Mesurer le recouvrement (Jaccard). Les écarts doivent s'expliquer (top-N creux, règles réactivées).

**Critères de fin :**
- `python -m pipeline.build` reconstruit tout depuis zéro.
- La taille totale des artefacts est **< 60 Mo** (contre 618 Mo).
- Le temps de chargement de l'app et la RAM sont mesurés et notés dans le README (avant / après).
- Les 4 règles de `smart_weights` sont actives, avec un test qui le prouve sur un vrai film non anglophone (ex. *Parasite*).

---

## Phase 2 — Évaluation hors-ligne (2–3 jours)

**But :** remplacer le « jugé à l'œil » par des métriques. C'est la phase qui produit les KPIs du CV.

### 2.1 Protocole

- **Données :** MovieLens `ml-latest-small` (ou `ml-25m` pour plus de robustesse), relié aux `tmdb_id` du catalogue via `links.csv`. Ne garder que les films présents dans le catalogue.
- **Positifs :** une note ≥ 4.
- **Tâche « item-to-item »** (celle de l'app) : pour chaque utilisateur, on prend un film aimé comme **requête**. Les autres films aimés de cet utilisateur, dans la partie test, sont les **pertinents**. On mesure si le moteur les classe dans son top-k.
- **Découpage temporel** par utilisateur : les 80 % de notes les plus anciennes servent au train, les 20 % plus récentes au test. Ça évite de prédire le passé avec le futur.
- **Point critique, la fuite de données :** le signal `collab` doit être **recalculé uniquement sur la partie train**. Sinon, il « connaît » les réponses et les scores sont gonflés.
- **Validation / test :** 10 % des utilisateurs servent de validation pour régler les poids. Le reste sert de test, utilisé **une seule fois** pour les chiffres finaux.

### 2.2 Métriques

| Famille | Métriques |
| :-- | :-- |
| Pertinence | Precision@k, Recall@k, **NDCG@10**, HitRate@10, MRR (k = 4, 10) |
| Couverture | % du catalogue apparaissant au moins une fois dans un top-10 |
| Diversité | Dissimilarité moyenne intra-liste (sur les embeddings SBERT) |
| Biais de popularité | Popularité moyenne des films recommandés vs celle du catalogue |
| Cold start | Toutes les métriques, calculées séparément sur les films sans données collaboratives (43 % du catalogue) |

Donner un **intervalle de confiance** (bootstrap sur les utilisateurs, 1 000 tirages) pour chaque chiffre clé.

### 2.3 Tâches

- [x] **2.3.1** `evaluation/dataset.py` : chargement de MovieLens, jointure avec le catalogue, découpage train / validation / test, positifs.
- [x] **2.3.2** `evaluation/metrics.py` : métriques, avec tests unitaires sur des cas calculés à la main.
- [x] **2.3.3** `evaluation/baselines.py`. Baselines à comparer :
  - Popularité (top global, hors film requête)
  - Aléatoire
  - SBERT seul
  - Collaboratif seul
  - Hybride à poids uniformes
  - Hybride à poids de base (`BASE_WEIGHTS`) sans règles
  - **Hybride actuel** (poids de base + règles adaptatives)
- [x] **2.3.4 Ablation.** Retirer chaque signal à tour de rôle et mesurer la perte de NDCG@10. Faire de même pour chaque règle de `smart_weights`, pour vérifier qu'elle aide vraiment.
- [x] **2.3.5 Optimisation des poids.** Recherche des 8 poids sur la validation (Optuna, environ 200 essais, objectif NDCG@10). Comparer poids manuels et poids optimisés sur le test.
- [x] **2.3.6 Rapport.** `python -m evaluation.run` produit `reports/eval.md` (tableaux et graphique d'ablation) et `reports/eval.json` (résultats bruts, graine aléatoire, versions). Ajouter un résumé dans le README.

**Critères de fin :**
- Tableau « baselines vs hybride » avec intervalles de confiance.
- Graphique d'ablation.
- Les chiffres du README et du CV viennent tous de `reports/eval.json`.

**Résultat possible et acceptable :** l'hybride peut ne pas battre le collaboratif seul sur les films avec beaucoup de notes, mais le battre largement sur le cold start. C'est un résultat honnête et intéressant à présenter : il justifie l'approche hybride.

---

## Phase 3 — Qualité du code & CI (1–2 jours)

- [x] **3.1 Découper `app.py`** :
  - `app.py` : point d'entrée de moins de 100 lignes
  - `ui/styles.css` : le CSS actuel (chargé via `st.markdown`)
  - `ui/components.py` : carte film, radar, en-tête
  - `cinematch/tmdb.py` : client TMDB avec timeout, nouvelles tentatives et cache
- [x] **3.2 Configuration déclarative.** Remplacer les nombres magiques de `smart_weights` (0.15, 0.30, seuil 3000, bonus 0.2/3) par une `WeightConfig` (dataclass, ou YAML chargé par `config.py`). Les règles deviennent des données qu'on peut tester et optimiser (phase 2.3.5).
- [x] **3.3 Corriger le bonus du cold start.** Il ajoute une constante `0.2 / 3` au lieu de redistribuer le poids **réel** de `Public` (qui peut valoir 0.30 ou 0.45 après les règles). Redistribuer `weights["Public"]` lui-même, et ajuster le test.
- [x] **3.4 Typage et lint.** `pyproject.toml` (remplace les `requirements*.txt`, avec dépendances optionnelles `[dev]` et `[pipeline]`), `ruff` (lint + format), `mypy --strict` sur `cinematch/`, `pre-commit`.
- [x] **3.5 Tests :**
  - Tests du moteur sur le nouveau format d'artefacts (embeddings + creux)
  - Tests du client TMDB avec `responses` ou `requests-mock` (succès, 404, timeout, pas de clé)
  - Tests de l'UI via `streamlit.testing.v1.AppTest` sur un mini-jeu d'artefacts synthétiques : l'app démarre, la recherche renvoie 4 cartes
  - Un test d'intégration `@pytest.mark.slow` sur les vrais artefacts
  - Mesurer la couverture avec `pytest-cov` (objectif ≥ 85 % sur `cinematch/`)
- [x] **3.6 CI GitHub Actions :** lint, mypy et tests sur chaque push et PR (Python 3.12 et 3.14). Badge dans le README.
- [x] **3.7 Benchmark.** `scripts/bench.py` mesure la latence p50 / p95 de `recommend()` sur 1 000 requêtes et le temps de chargement des artefacts. Le chiffre de « 0,2 ms » du README doit venir de ce script.

**Critère de fin :** CI verte, couverture mesurée, `app.py` < 100 lignes.

---

## Phase 4 — Fonctionnalités produit & UI (2–4 jours)

Par ordre de priorité :

- [x] **4.1 Explication par recommandation.** Sous chaque carte, indiquer les 2 signaux qui ont le plus contribué (`w_k × M_k[i,j]`), par exemple « Même réalisateur · ambiance proche ». `recommend()` renvoie déjà les poids ; il faut ajouter les contributions par signal. C'est le meilleur argument de l'approche explicable, bien plus parlant que le radar global.
- [x] **4.2 Curseurs de poids.** Dans un panneau « Mode avancé », permettre de régler les 8 poids, avec recalcul instantané et bouton « Réinitialiser ».
- [x] **4.3 Diversité (MMR).** Ajouter un curseur « Diversité » (λ). Re-classer les ~50 meilleurs candidats par Maximal Marginal Relevance pour éviter 4 films de la même saga. Mesurer l'effet sur la diversité et le NDCG (phase 2).
- [x] **4.4 Requête multi-films.** « J'ai aimé A, B et C » : agréger les scores des films requêtes (moyenne des lignes de similarité).
- [x] **4.5 Filtres.** Décennie, genre, langue, durée, appliqués après le calcul du score.
- [x] **4.6 Nombre de résultats.** Choix entre 4, 8 et 12, avec pagination.
- [x] **4.7 Langue de l'UI.** Bascule FR / EN, avec les libellés centralisés dans un dictionnaire. Les noms de signaux affichés (`Histoire`, `Sujets`…) passent en EN par défaut, pour être cohérents avec le README.
- [x] **4.8 Détails.** Lien vers la fiche TMDB, bande-annonce si disponible, état vide propre si TMDB ne répond pas, accessibilité (contraste, attribut `alt` sur les affiches).

Le design actuel (thème sombre, radar, cartes) est solide : **pas de refonte**, seulement ces ajouts.

---

## Phase 5 — Passage à l'échelle (optionnel, 3–5 jours)

À faire seulement si les phases 0 à 4 sont terminées. C'est un bon sujet d'entretien (« comment ça marche avec 100 000 films ? »).

- [ ] **5.1 Catalogue élargi.** Tous les films MovieLens 25M liés à TMDB (environ 60 000), ou un export TMDB complet filtré par nombre de votes.
- [ ] **5.2 Index ANN.** FAISS (HNSW ou IVF) sur les embeddings pour obtenir les candidats (top 200), puis re-classement hybride sur ces candidats seulement. Mesurer latence et rappel de l'ANN par rapport à la recherche exacte.
- [ ] **5.3 Collaboratif moderne.** Remplacer la similarité item-item par un modèle appris (ALS via `implicit`, ou le *dual tower* déjà esquissé dans `hybrid-dual-tower-movie-reco-system.ipynb`), évalué avec le protocole de la phase 2.
- [ ] **5.4 Apprendre le mélange.** Remplacer les règles par un modèle learning-to-rank (LightGBM `LGBMRanker`) qui prend en entrée les 8 similarités et les attributs du film. Comparer avec les règles manuelles et les poids optimisés.
- [ ] **5.5 API.** Exposer le moteur via FastAPI (`/recommend?tmdb_id=…&k=…`), avec Dockerfile, healthcheck et tests. L'app Streamlit devient un client de l'API.

---

## Phase 6 — Valorisation (½ jour)

- [ ] **6.1 README.** Ajouter un GIF de démo en tête, la section « Evaluation » (tableau baselines + ablation), et mettre à jour « Limitations » (cocher ce qui est fait).
- [ ] **6.2 Portfolio.** Mettre à jour la fiche du projet dans `Victor_MICHELON_Portfolio.github.io` avec les nouveaux chiffres.
- [ ] **6.3 Article court (optionnel).** « Construire un recommandeur hybride explicable et l'évaluer honnêtement », sur LinkedIn ou Medium, à partir de `reports/eval.md`.
- [ ] **6.4 CV.** Voir ci-dessous.

### KPIs pour le CV

N'utiliser **que des chiffres produits par `reports/eval.json` et `scripts/bench.py`**. Les crochets sont à remplacer par les vraies valeurs.

**Déjà vérifiables aujourd'hui :**
- 8 signaux de similarité hybrides (contenu sémantique, métadonnées, collaboratif), 6 203 films.
- Gestion du cold start pour 43 % du catalogue sans données collaboratives.

**Après les phases 1 à 3 :**
- « Évalué hors-ligne sur [N] utilisateurs MovieLens : NDCG@10 de [X], soit **+[Y] %** vs baseline popularité et **+[Z] %** vs SBERT seul. »
- « Sur les films en cold start : **+[W] %** de HitRate@10 vs filtrage collaboratif seul. »
- « Optimisation bayésienne des poids (Optuna) : **+[V] points** de NDCG@10 vs pondération manuelle. »
- « Artefacts réduits de 618 Mo à **[M] Mo** (−[P] %) grâce aux embeddings et matrices creuses ; latence p95 de **[L] ms**. »
- « Couverture de tests de [C] %, CI GitHub Actions. »

**Exemple de ligne CV :**
> **CineMatch** — Recommandeur de films hybride et explicable (Python, Sentence-Transformers, SciPy, Optuna, Streamlit). 8 signaux avec pondération adaptative et gestion du cold start (43 % du catalogue). Protocole d'évaluation sans fuite sur MovieLens : NDCG@10 **+[Y] %** vs baseline popularité, ablation par signal. Pipeline reproductible, artefacts ÷[R], CI et [C] % de couverture. Démo en ligne.

---

## Ordre d'exécution conseillé

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 6 (1er jet du CV)
                 │
                 └──► Phase 3 (en parallèle de 2)
                              └──► Phase 4 ──► Phase 5 (optionnel)
```

Faire une branche et une PR par sous-phase (par exemple `feat/sparse-artifacts`, `feat/offline-eval`), chacune avec ses tests, pour que l'historique Git montre une démarche structurée.

## Risques & points d'attention

| Risque | Mitigation |
| :-- | :-- |
| Le code de `vibe` / `collab` est introuvable | Reconstruire ces signaux proprement (par exemple `vibe` = embeddings de tagline + keywords), documenter le changement, comparer à l'ancien avec le Jaccard de la phase 1.2.9 |
| Fuite de données dans l'évaluation | `collab` recalculé sur le train uniquement, jeu de test utilisé une seule fois, graine fixée |
| L'hybride ne bat pas toutes les baselines | Le présenter tel quel : analyser par segment (cold start, popularité) et en tirer la justification de l'approche |
| Faible recouvrement MovieLens ↔ catalogue (films récents) | Rapporter la taille effective du jeu d'évaluation ; évaluer les films récents à part, ou les exclure explicitement |
| Mémoire sur Streamlit Cloud | Réglé par la phase 1 (< 60 Mo d'artefacts) |
| Conditions d'utilisation TMDB et MovieLens | Ne pas redistribuer les notes brutes ; garder l'attribution actuelle |
