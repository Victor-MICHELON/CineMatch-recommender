"""Benchmark : temps de chargement des artefacts et latence de `recommend()`.

Usage : python scripts/bench.py [--n 1000]  ->  reports/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cinematch import load_artifacts, recommend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.perf_counter()
    art = load_artifacts(ROOT / "artifacts")
    load_s = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    queries = rng.integers(0, art.n, size=args.n)
    for q in queries[:20]:  # échauffement
        recommend(art, int(q))
    lat = []
    for q in queries:
        t = time.perf_counter()
        recommend(art, int(q), k=4)
        lat.append((time.perf_counter() - t) * 1000)
    lat_mmr = []
    for q in queries[: args.n // 10]:
        t = time.perf_counter()
        recommend(art, int(q), k=4, diversity=0.5)
        lat_mmr.append((time.perf_counter() - t) * 1000)

    result = {
        "n_queries": args.n,
        "n_movies": art.n,
        "load_artifacts_s": round(load_s, 3),
        "recommend_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        },
        "recommend_mmr_ms": {
            "p50": round(float(np.percentile(lat_mmr, 50)), 2),
            "p95": round(float(np.percentile(lat_mmr, 95)), 2),
        },
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }
    out = ROOT / "reports" / "bench.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
