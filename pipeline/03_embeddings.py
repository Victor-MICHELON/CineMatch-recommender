"""Étape 03 — Signaux sémantiques `sbert` (histoire) et `vibe` (ambiance).

Les deux sont des embeddings Sentence-Transformers normalisés (cosinus = produit
scalaire), stockés en float16 : 6 203 × 384 ≈ 4,8 Mo chacun.

- `sbert` : `all-MiniLM-L6-v2` sur « titre. synopsis » (notebook d'origine).
- `vibe`  : le code d'origine n'était pas versionné. Reconstruction : même modèle
  sur « accroche. genres. mots-clés », c'est-à-dire le ton et l'univers du film
  plutôt que son intrigue. Voir `reports/regression.md` pour la comparaison
  avec l'ancien signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common import ARTIFACTS_DIR

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def plot_text(cat: pd.DataFrame) -> list[str]:
    return (cat["title"] + ". " + cat["overview"]).tolist()


def vibe_text(cat: pd.DataFrame) -> list[str]:
    parts = []
    for tagline, genres, keywords in zip(cat["tagline"], cat["genres_list"], cat["keywords_list"], strict=True):
        parts.append(". ".join(p for p in (tagline, ", ".join(genres), ", ".join(list(keywords)[:15])) if p))
    return parts


def encode(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # dépendance lourde, pipeline seulement

    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float16)


def main() -> None:
    cat = pd.read_parquet(ARTIFACTS_DIR / "catalog.parquet")
    for signal, texts in (("sbert", plot_text(cat)), ("vibe", vibe_text(cat))):
        emb = encode(texts)
        np.save(ARTIFACTS_DIR / f"emb_{signal}.npy", emb, allow_pickle=False)
        print(f"emb_{signal}.npy : {emb.shape}, {emb.nbytes / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
