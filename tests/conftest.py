"""Mini-jeu d'artefacts synthétiques (8 films), sans réseau ni gros fichiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cinematch.config import EMBEDDING_SIGNALS, SPARSE_SIGNALS
from cinematch.data import CATALOG, Artifacts, embedding_file, factor_file, load_artifacts, write_manifest

N = 8


def make_catalog(n: int = N) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(100, 100 + n),
            "title": [f"Movie {chr(65 + i)}" for i in range(n)],
            "year": [2000 + i for i in range(n)],
            "director": ["Dir A", "Dir A", "Dir B", "", "Dir C", "Dir C", "Dir D", "Dir E"][:n],
            "genres_list": [
                ["Drama"],
                ["Drama"],
                ["Horror"],
                ["Animation"],
                ["Comedy"],
                ["Drama"],
                ["Action"],
                ["Drama"],
            ][:n],
            "keywords_list": [["k"]] * n,
            "cast_list": [["actor"]] * n,
            "countries_list": [["US"]] * n,
            "original_language": ["en", "en", "en", "en", "ko", "en", "fr", "en"][:n],
            "overview": ["A story."] * n,
            "tagline": [""] * n,
            "popularity": np.linspace(1, 100, n).astype("float32"),
            "vote_average": [7.0] * n,
            "vote_count": [100, 5000, 50, 10, 20, 30, 40, 4000][:n],
            "runtime": [90, 100, 110, 120, 130, 140, 150, 160][:n],
            "movielens_id": pd.array([1, 2, 3, None, 5, 6, 7, 8][:n], dtype="Int64"),
        }
    )


def normalized(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def make_artifacts(n: int = N, seed: int = 0) -> Artifacts:
    rng = np.random.default_rng(seed)
    embeddings = {s: normalized(rng.normal(size=(n, 16))).astype(np.float32) for s in EMBEDDING_SIGNALS}
    factors = {}
    for s in SPARSE_SIGNALS:
        dense = (rng.random((n, 12)) > 0.6).astype(np.float32)
        if s == "collab":
            dense[3] = 0  # film 3 : cold start
        factors[s] = sp.csr_matrix(normalized(dense), dtype=np.float32)
    return Artifacts(make_catalog(n), embeddings, factors)


def write_artifacts(art: Artifacts, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    art.catalog.to_parquet(directory / CATALOG, index=False)
    for s in EMBEDDING_SIGNALS:
        np.save(directory / embedding_file(s), art.embeddings[s].astype(np.float16), allow_pickle=False)
    for s in SPARSE_SIGNALS:
        sp.save_npz(directory / factor_file(s), art.factors[s])
    write_manifest(directory, {"format_version": 1, "n_movies": art.n})


@pytest.fixture
def art() -> Artifacts:
    return make_artifacts()


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    write_artifacts(make_artifacts(), d)
    return d


@pytest.fixture
def loaded(artifacts_dir: Path) -> Artifacts:
    return load_artifacts(artifacts_dir)
