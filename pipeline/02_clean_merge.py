"""Étape 02 — Nettoyage, jointure MovieLens et écriture de `catalog.parquet`.

Le catalogue contient toutes les colonnes utiles au moteur et à l'UI, dont
`original_language` et `vote_count` (règles adaptatives) et `movielens_id`
(signal collaboratif, évaluation).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pipeline.common import ARTIFACTS_DIR, CATALOG_IDS, TMDB_JSONL, ensure_movielens

N_CAST = 4  # acteurs retenus pour le signal `credits` (comme le notebook d'origine)


def load_tmdb() -> pd.DataFrame:
    rows = {}
    with TMDB_JSONL.open() as f:
        for line in f:
            rec = json.loads(line)
            rows[rec["requested_id"]] = rec["movie"]
    ids = pd.read_csv(CATALOG_IDS)
    movies = []
    for tmdb_id, source in zip(ids["tmdb_id"], ids["source"], strict=True):
        movie = rows.get(int(tmdb_id))
        if movie is None:
            continue  # absent du cache ou supprimé de TMDB
        movie = dict(movie)
        movie["source"] = source
        movies.append(movie)
    print(f"Catalogue : {len(movies)} films sur {len(ids)} identifiants ({len(ids) - len(movies)} introuvables)")
    return pd.DataFrame(movies)


def build_catalog(raw: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(raw["release_date"], errors="coerce")
    cat = pd.DataFrame(
        {
            "id": raw["id"].astype("int64"),
            "imdb_id": raw["imdb_id"].fillna("").astype(str),
            "title": raw["title"].fillna("").astype(str),
            "original_title": raw["original_title"].fillna("").astype(str),
            "year": date.dt.year.fillna(0).astype("int64"),
            "director": raw["directors"].map(lambda d: d[0] if d else ""),
            "genres_list": raw["genres"].map(lambda g: [x["name"] for x in g or []]),
            "keywords_list": raw["keywords"].map(lambda k: list(k or [])),
            "cast_list": raw["cast"].map(lambda c: list(c or [])[:N_CAST]),
            "countries_list": raw["production_countries"].map(lambda c: [x["iso_3166_1"] for x in c or []]),
            "original_language": raw["original_language"].fillna("").astype(str),
            "overview": raw["overview"].fillna("").astype(str),
            "tagline": raw["tagline"].fillna("").astype(str),
            "popularity": pd.to_numeric(raw["popularity"], errors="coerce").fillna(0.0).astype("float32"),
            "vote_average": pd.to_numeric(raw["vote_average"], errors="coerce").fillna(0.0).astype("float32"),
            "vote_count": pd.to_numeric(raw["vote_count"], errors="coerce").fillna(0).astype("int64"),
            "runtime": pd.to_numeric(raw["runtime"], errors="coerce").fillna(0).astype("int64"),
            "source": raw["source"].astype(str),
        }
    )
    tmdb_to_ml = (
        links.dropna(subset=["tmdbId"])
        .astype({"tmdbId": "int64"})
        .drop_duplicates("tmdbId")
        .set_index("tmdbId")["movieId"]
    )
    cat["movielens_id"] = cat["id"].map(tmdb_to_ml).astype("Int64")
    return cat.reset_index(drop=True)


def main() -> None:
    raw = load_tmdb()
    links = pd.read_csv(ensure_movielens() / "links.csv")
    cat = build_catalog(raw, links)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    cat.to_parquet(ARTIFACTS_DIR / "catalog.parquet", index=False)
    print(
        f"catalog.parquet : {len(cat)} films, "
        f"{cat['movielens_id'].notna().mean():.1%} reliés à MovieLens, "
        f"{(cat['original_language'] != 'en').mean():.1%} non anglophones, "
        f"{(cat['vote_count'] > 3000).mean():.1%} avec > 3000 votes, "
        f"années {cat.loc[cat['year'] > 0, 'year'].min()}–{cat['year'].max()} "
        f"(médiane {int(np.median(cat.loc[cat['year'] > 0, 'year']))})"
    )


if __name__ == "__main__":
    main()
