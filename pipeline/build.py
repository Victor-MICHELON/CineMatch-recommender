"""Reconstruit tous les artefacts depuis les sources brutes : `python -m pipeline.build`.

Options :
  --skip-fetch   réutilise le cache TMDB sans interroger l'API
  --zip          produit aussi `artifacts.zip` (à publier en GitHub Release)
"""

from __future__ import annotations

import argparse
import importlib
import platform
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version

import pandas as pd

from cinematch.config import ContextConfig
from cinematch.data import MANIFEST, write_manifest
from pipeline.common import ARTIFACTS_DIR, MOVIELENS_URL

STEPS = ("01_fetch_tmdb", "02_clean_merge", "03_embeddings", "04_content_sims", "05_collab")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    for step in STEPS:
        if args.skip_fetch and step == "01_fetch_tmdb":
            continue
        t0 = time.perf_counter()
        print(f"\n=== {step}")
        importlib.import_module(f"pipeline.{step}").main()
        print(f"({time.perf_counter() - t0:.1f} s)")

    cat = pd.read_parquet(ARTIFACTS_DIR / "catalog.parquet")
    ctx = ContextConfig()
    manifest = write_manifest(
        ARTIFACTS_DIR,
        {
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_movies": len(cat),
            "context": {"pop_sigma": ctx.pop_sigma, "year_sigma": ctx.year_sigma},
            "sources": {"tmdb": "API TMDB /movie/{id}", "movielens": MOVIELENS_URL},
            "versions": {
                "python": platform.python_version(),
                **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "sentence-transformers")},
            },
        },
    )
    files: dict[str, dict[str, int]] = manifest["files"]  # type: ignore[assignment]
    total = sum(f["bytes"] for f in files.values())
    print(f"\nArtefacts : {len(files)} fichiers, {total / 1e6:.1f} Mo -> {ARTIFACTS_DIR}")

    if args.zip:
        archive = ARTIFACTS_DIR.parent / "artifacts.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in [*files, MANIFEST]:
                zf.write(ARTIFACTS_DIR / name, arcname=name)
        print(f"Archive : {archive} ({archive.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
