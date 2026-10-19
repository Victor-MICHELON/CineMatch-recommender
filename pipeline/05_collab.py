"""Étape 05 — Signal collaboratif `collab` (MovieLens `ml-latest-small`).

Similarité item-item cosinus sur les vecteurs de notes (film × utilisateur).
Facteur stocké : la matrice film × utilisateur à lignes normalisées, alignée
sur le catalogue ; un film sans note a une ligne vide (cold start).

`collab_factor` (dans `pipeline.common`) est réutilisée par l'évaluation pour recalculer le signal sur
la partie train uniquement (pas de fuite).
"""

from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

from pipeline.common import ARTIFACTS_DIR, collab_factor, ensure_movielens


def main() -> None:
    cat = pd.read_parquet(ARTIFACTS_DIR / "catalog.parquet")
    ratings = pd.read_csv(ensure_movielens() / "ratings.csv")
    f = collab_factor(ratings, cat["movielens_id"])
    sp.save_npz(ARTIFACTS_DIR / "fac_collab.npz", f)
    cold = (f.getnnz(axis=1) == 0).mean()
    print(f"fac_collab.npz : {f.shape}, {f.nnz} notes, {cold:.1%} du catalogue sans donnée collaborative")


if __name__ == "__main__":
    main()
