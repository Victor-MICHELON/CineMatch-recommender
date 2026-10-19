"""Chargement des artefacts (catalogue Parquet, embeddings .npy, facteurs creux .npz).

Chaque signal de similarité est stocké sous forme factorisée, jamais en matrice
n × n : la ligne `i` de la similarité vaut `F @ F[i]`, avec `F` à lignes
normalisées (L2). On obtient exactement le cosinus, pour quelques Mo au lieu de
centaines.

- `sbert`, `vibe` : embeddings denses (n × 384, float16 sur disque) ;
- `keywords`, `credits`, `genre`, `origin`, `collab` : matrices creuses (n × vocabulaire) ;
- `context` : fonction analytique de l'année et de la popularité, calculée à la volée.

Aucun pickle : Parquet / NumPy (`allow_pickle=False`) / SciPy, et un
`manifest.json` dont les empreintes SHA-256 sont vérifiées au chargement.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from numpy.typing import NDArray

from cinematch.config import EMBEDDING_SIGNALS, SIGNALS, SPARSE_SIGNALS, ContextConfig

MANIFEST = "manifest.json"
CATALOG = "catalog.parquet"
LIST_COLUMNS = ("genres_list", "keywords_list", "cast_list", "countries_list")
DEFAULT_ARTIFACTS_URL = (
    "https://github.com/Victor-MICHELON/CineMatch-recommender/releases/download/artifacts-v1/artifacts.zip"
)

FloatArray = NDArray[np.float32]


class ArtifactError(RuntimeError):
    """Artefacts absents, incomplets ou corrompus."""


def embedding_file(signal: str) -> str:
    return f"emb_{signal}.npy"


def factor_file(signal: str) -> str:
    return f"fac_{signal}.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Artifacts:
    """Catalogue + représentations factorisées des 8 signaux."""

    catalog: pd.DataFrame
    embeddings: dict[str, FloatArray]
    factors: dict[str, sp.csr_matrix]
    context: ContextConfig = field(default_factory=ContextConfig)
    manifest: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.catalog)
        for s in EMBEDDING_SIGNALS:
            if self.embeddings[s].shape[0] != n:
                raise ArtifactError(f"embedding {s} : {self.embeddings[s].shape[0]} lignes pour {n} films")
        for s in SPARSE_SIGNALS:
            if self.factors[s].shape[0] != n:
                raise ArtifactError(f"facteur {s} : {self.factors[s].shape[0]} lignes pour {n} films")
        self._year = self.catalog["year"].to_numpy(dtype=np.float32)
        self._pop = self.catalog["popularity"].to_numpy(dtype=np.float32)
        self._has_collab = np.asarray(self.factors["collab"].getnnz(axis=1) > 0)

    @property
    def n(self) -> int:
        return len(self.catalog)

    def has_collab(self, idx: int) -> bool:
        return bool(self._has_collab[idx])

    @property
    def collab_coverage(self) -> float:
        return float(self._has_collab.mean())

    def similarity(self, signal: str, idx: int) -> FloatArray:
        """Ligne `idx` de la matrice de similarité du signal (vecteur de taille n)."""
        if signal in EMBEDDING_SIGNALS:
            emb = self.embeddings[signal]
            sim: FloatArray = emb @ emb[idx]
            return sim
        if signal in SPARSE_SIGNALS:
            fac = self.factors[signal]
            row = fac[idx].toarray().ravel()
            prod: FloatArray = np.asarray(fac @ row, dtype=np.float32)
            return prod
        if signal == "context":
            pop = np.exp(-np.abs(self._pop - self._pop[idx]) / self.context.pop_sigma)
            year = np.exp(-np.abs(self._year - self._year[idx]) / self.context.year_sigma)
            ctx: FloatArray = ((pop + year) / 2).astype(np.float32)
            return ctx
        raise KeyError(signal)

    def similarities(self, idx: int) -> FloatArray:
        """Les 8 lignes de similarité, empilées dans l'ordre de `SIGNALS` (8 × n)."""
        stack: FloatArray = np.stack([self.similarity(s, idx) for s in SIGNALS])
        return stack

    def with_factor(self, signal: str, factor: sp.csr_matrix) -> Artifacts:
        """Copie avec un facteur remplacé (ex. `collab` recalculé sur le train)."""
        return replace(self, factors={**self.factors, signal: factor})


def load_artifacts(directory: Path, verify: bool = True) -> Artifacts:
    """Charge les artefacts d'un dossier, en vérifiant le manifeste."""
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise ArtifactError(f"{manifest_path} introuvable : lancer `python -m pipeline.build`")
    manifest = json.loads(manifest_path.read_text())
    files: dict[str, dict[str, object]] = manifest["files"]
    if verify:
        for name, meta in files.items():
            path = directory / name
            if not path.exists():
                raise ArtifactError(f"artefact manquant : {name}")
            if sha256(path) != meta["sha256"]:
                raise ArtifactError(f"empreinte SHA-256 invalide : {name}")

    catalog = pd.read_parquet(directory / CATALOG)
    for col in LIST_COLUMNS:
        if col in catalog:
            catalog[col] = catalog[col].map(lambda v: [str(x) for x in v] if v is not None else [])
    embeddings = {
        s: np.load(directory / embedding_file(s), allow_pickle=False).astype(np.float32) for s in EMBEDDING_SIGNALS
    }
    factors = {s: sp.csr_matrix(sp.load_npz(directory / factor_file(s)), dtype=np.float32) for s in SPARSE_SIGNALS}
    context = ContextConfig(**manifest.get("context", {}))
    return Artifacts(catalog, embeddings, factors, context, manifest)


def ensure_artifacts(directory: Path, url: str | None = None) -> Path:
    """Télécharge et décompresse les artefacts publiés si le dossier est vide."""
    if (directory / MANIFEST).exists():
        return directory
    url = url or os.environ.get("CINEMATCH_ARTIFACTS_URL") or DEFAULT_ARTIFACTS_URL
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            # Archive à plat uniquement : on refuse tout chemin qui sortirait du dossier
            if Path(member).name != member:
                raise ArtifactError(f"chemin inattendu dans l'archive : {member}")
        zf.extractall(directory)
    return directory


def write_manifest(directory: Path, extra: dict[str, object]) -> dict[str, object]:
    """Écrit `manifest.json` (empreinte et taille de chaque fichier + métadonnées)."""
    names = [CATALOG] + [embedding_file(s) for s in EMBEDDING_SIGNALS] + [factor_file(s) for s in SPARSE_SIGNALS]
    files = {n: {"sha256": sha256(directory / n), "bytes": (directory / n).stat().st_size} for n in names}
    manifest = {**extra, "files": files}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
