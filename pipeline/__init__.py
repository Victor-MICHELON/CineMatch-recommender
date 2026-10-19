"""Pipeline de données de CineMatch : sources brutes -> `artifacts/`.

Étapes (chacune exécutable seule, ou toutes via `python -m pipeline.build`) :

01_fetch_tmdb    métadonnées TMDB des films de `catalog_ids.csv` (cache JSONL)
02_clean_merge   nettoyage + jointure MovieLens -> catalog.parquet
03_embeddings    SBERT (histoire) et vibe -> embeddings .npy normalisés
04_content_sims  keywords, credits, genre, origin -> facteurs creux .npz
05_collab        MovieLens -> facteurs collaboratifs creux .npz
"""
