"""Étape 01 — Métadonnées TMDB de tous les films du catalogue.

Remplace les CSV Kaggle (TMDB 5000 + export des films récents) par l'API TMDB,
interrogée pour chaque `tmdb_id` de `pipeline/catalog_ids.csv`. Les réponses
(réduites aux champs utiles) sont mises en cache dans `data/raw/tmdb/movies.jsonl` :
une relance ne re-télécharge que les films manquants.

Clé : variable d'environnement `TMDB_API_KEY`, ou `.streamlit/secrets.toml`.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from pipeline.common import CATALOG_IDS, ROOT, TMDB_JSONL

API = "https://api.themoviedb.org/3/movie/{id}"
KEEP = (
    "id",
    "imdb_id",
    "title",
    "original_title",
    "original_language",
    "overview",
    "tagline",
    "release_date",
    "popularity",
    "vote_average",
    "vote_count",
    "runtime",
    "genres",
    "production_countries",
    "spoken_languages",
)
N_CAST = 10


def api_key() -> str:
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        key = tomllib.loads(secrets.read_text()).get("TMDB_API_KEY")
        if key:
            return str(key)
    raise SystemExit("TMDB_API_KEY introuvable (variable d'environnement ou .streamlit/secrets.toml)")


def fetch_one(session: requests.Session, tmdb_id: int, key: str) -> dict | None:
    """Une fiche film réduite, ou None si le film n'existe plus sur TMDB."""
    for attempt in range(5):
        resp = session.get(
            API.format(id=tmdb_id),
            params={"api_key": key, "append_to_response": "keywords,credits"},
            timeout=15,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 2**attempt)))
            continue
        if resp.status_code >= 500:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        raw = resp.json()
        movie = {k: raw.get(k) for k in KEEP}
        movie["keywords"] = [kw["name"] for kw in raw.get("keywords", {}).get("keywords", [])]
        credits = raw.get("credits", {})
        movie["cast"] = [c["name"] for c in sorted(credits.get("cast", []), key=lambda c: c.get("order", 0))[:N_CAST]]
        movie["directors"] = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
        return movie
    raise RuntimeError(f"TMDB : échec répété pour le film {tmdb_id}")


def main() -> None:
    ids = pd.read_csv(CATALOG_IDS)["tmdb_id"].astype(int).tolist()
    TMDB_JSONL.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if TMDB_JSONL.exists():
        with TMDB_JSONL.open() as f:
            done = {json.loads(line)["requested_id"] for line in f}
    todo = [i for i in ids if i not in done]
    print(f"TMDB : {len(done)} films en cache, {len(todo)} à récupérer")

    key = api_key()
    session = requests.Session()
    missing = 0
    with TMDB_JSONL.open("a") as out, ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, session, i, key): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            tmdb_id = futures[fut]
            movie = fut.result()
            missing += movie is None
            out.write(json.dumps({"requested_id": tmdb_id, "movie": movie}, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"TMDB : terminé ({missing} films introuvables sur cette passe)")


if __name__ == "__main__":
    main()
