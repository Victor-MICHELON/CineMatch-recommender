"""Chemins et utilitaires partagés par les étapes du pipeline."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
CATALOG_IDS = Path(__file__).resolve().parent / "catalog_ids.csv"

TMDB_JSONL = RAW_DIR / "tmdb" / "movies.jsonl"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
MOVIELENS_DIR = RAW_DIR / "ml-latest-small"


def _download_movielens() -> bytes:
    try:
        resp = requests.get(MOVIELENS_URL, timeout=60)
    except requests.exceptions.SSLError as exc:
        # files.grouplens.org ne sert pas toujours sa chaîne de certificats complète,
        # que `certifi` ne sait pas reconstituer (curl, lui, y arrive via le système).
        raise SystemExit(
            "Téléchargement MovieLens impossible (certificat TLS). Récupérer l'archive avec :\n"
            f"  curl -L -o {RAW_DIR / 'ml-latest-small.zip'} {MOVIELENS_URL}\n"
            "puis relancer : son empreinte SHA-256 sera vérifiée."
        ) from exc
    resp.raise_for_status()
    return resp.content


def ensure_movielens() -> Path:
    """Télécharge, vérifie et décompresse MovieLens `ml-latest-small` si besoin (~1 Mo)."""
    if not (MOVIELENS_DIR / "ratings.csv").exists():
        archive = RAW_DIR / "ml-latest-small.zip"
        if archive.exists():
            content = archive.read_bytes()
        else:
            print(f"Téléchargement de {MOVIELENS_URL}")
            content = _download_movielens()
        digest = hashlib.sha256(content).hexdigest()
        if digest != MOVIELENS_SHA256:
            raise RuntimeError(f"MovieLens : empreinte inattendue {digest}")
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(RAW_DIR)
    return MOVIELENS_DIR


def collab_factor(ratings: pd.DataFrame, movielens_ids: pd.Series) -> sp.csr_matrix:
    """Facteur (n_films × n_utilisateurs) à lignes normalisées.

    `ratings` : colonnes `userId`, `movieId`, `rating`.
    `movielens_ids` : `movielens_id` de chaque film du catalogue (NA si absent).
    """
    pos = pd.Series(np.arange(len(movielens_ids)), index=movielens_ids.to_numpy())
    pos = pos[pos.index.notna()]
    pos = pos[~pos.index.duplicated()]
    r = ratings[ratings["movieId"].isin(pos.index)]
    users, user_idx = np.unique(r["userId"].to_numpy(), return_inverse=True)
    rows = pos.loc[r["movieId"].to_numpy()].to_numpy()
    mat = sp.csr_matrix(
        (r["rating"].to_numpy(dtype=np.float32), (rows, user_idx)),
        shape=(len(movielens_ids), len(users)),
    )
    return sp.csr_matrix(normalize(mat), dtype=np.float32)
