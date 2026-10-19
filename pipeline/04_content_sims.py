"""Étape 04 — Signaux de métadonnées `keywords`, `credits`, `genre`, `origin`.

Même construction que le notebook d'origine (sacs de mots `CountVectorizer`,
similarité cosinus), mais on stocke la matrice de caractéristiques à lignes
normalisées (creuse, quelques centaines de Ko) au lieu de la matrice de
similarité n × n : `F @ F[i]` redonne exactement la ligne de cosinus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from pipeline.common import ARTIFACTS_DIR


def sanitize(values: list[str]) -> list[str]:
    """« Science Fiction » -> « sciencefiction » : un token par valeur."""
    return [str(v).replace(" ", "").replace("-", "").lower() for v in values]


def soups(cat: pd.DataFrame) -> dict[str, list[str]]:
    keywords = [" ".join(sanitize(k)) for k in cat["keywords_list"]]
    # Le réalisateur est répété 3 fois pour peser plus que chaque acteur
    credits = [
        " ".join(sanitize([d] * 3 if d else []) + sanitize(c))
        for d, c in zip(cat["director"], cat["cast_list"], strict=True)
    ]
    genre = [" ".join(sanitize(g)) for g in cat["genres_list"]]
    origin = [
        " ".join([f"lang_{lang}"] + [f"country_{c.lower()}" for c in countries])
        for lang, countries in zip(cat["original_language"], cat["countries_list"], strict=True)
    ]
    return {"keywords": keywords, "credits": credits, "genre": genre, "origin": origin}


VECTORIZERS = {
    # min_df=2 : un mot-clé présent dans un seul film ne rapproche personne
    "keywords": dict(stop_words="english", min_df=2),
    "credits": dict(),
    "genre": dict(),
    "origin": dict(),
}


def factor(texts: list[str], **kwargs: object) -> sp.csr_matrix:
    vec = CountVectorizer(token_pattern=r"\S+", lowercase=False, **kwargs)  # noqa: S106  # type: ignore[arg-type]
    try:
        x = vec.fit_transform(texts)
    except ValueError:  # vocabulaire vide
        return sp.csr_matrix((len(texts), 0), dtype=np.float32)
    return sp.csr_matrix(normalize(x.astype(np.float32)), dtype=np.float32)


def main() -> None:
    cat = pd.read_parquet(ARTIFACTS_DIR / "catalog.parquet")
    for signal, texts in soups(cat).items():
        f = factor(texts, **VECTORIZERS[signal])
        sp.save_npz(ARTIFACTS_DIR / f"fac_{signal}.npz", f)
        empty = (f.getnnz(axis=1) == 0).mean()
        print(f"fac_{signal}.npz : {f.shape}, {f.nnz} non-zéros, {empty:.1%} de films sans donnée")


if __name__ == "__main__":
    main()
