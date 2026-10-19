"""Non-régression : ancien pickle dense (`recommender_data3.pkl`) vs nouveaux artefacts.

Usage : python scripts/compare_legacy.py [chemin/vers/recommender_data3.pkl]

Produit `reports/regression.md` et `reports/regression.json` :
- accord signal par signal (corrélation des lignes, recouvrement du top-20) ;
- Jaccard des top-10 de l'ancien et du nouveau moteur sur 50 films tirés au hasard ;
- temps de chargement et RAM maximale, avant / après.

Le pickle n'est lu que par ce script, sur un fichier local produit par le projet
lui-même (ne jamais charger un pickle d'origine inconnue).
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cinematch.config import SIGNALS  # noqa: E402
from cinematch.data import load_artifacts  # noqa: E402
from cinematch.engine import recommend  # noqa: E402

LEGACY = ROOT / "assets" / "recommender_data3.pkl"
REPORTS = ROOT / "reports"
N_FILMS, K, SEED = 50, 10, 0

# Ancien moteur (commit 1d008ad), recopié tel quel pour la comparaison
LEGACY_KEYS = {
    "Histoire": "sbert",
    "Sujets": "keywords",
    "Vibe": "vibe",
    "Genre": "genre",
    "Casting": "credits",
    "Origine": "origin",
    "Popularité": "context",
    "Public": "collab",
}
LEGACY_BASE = {
    "Histoire": 0.15,
    "Sujets": 0.25,
    "Vibe": 0.15,
    "Genre": 0.10,
    "Casting": 0.10,
    "Origine": 0.00,
    "Popularité": 0.05,
    "Public": 0.20,
}


def legacy_top(df: pd.DataFrame, mats: dict[str, np.ndarray], idx: int, k: int) -> list[int]:
    row = df.iloc[idx]
    genres = set(row["genres_list"]) if isinstance(row["genres_list"], list) else set()
    w = dict(LEGACY_BASE)
    if "Animation" in genres or any(ch.isdigit() for ch in row["title"]):
        w["Sujets"] += 0.15
        w["Public"] += 0.10
    if "Horror" in genres:
        w["Vibe"] += 0.20
        w["Sujets"] += 0.10
        w["Casting"] -= 0.05
    if mats["collab"][idx].sum() == 0:
        w["Public"] = 0.0
        for name in ("Histoire", "Sujets", "Vibe"):
            w[name] += 0.2 / 3
    scores = np.zeros(len(df), dtype=np.float32)
    for name, weight in w.items():
        if weight > 0:
            scores += weight * mats[LEGACY_KEYS[name]][idx].astype(np.float32)
    scores[idx] = -np.inf
    return [int(i) for i in np.argsort(-scores)[:k]]


def measure(code: str) -> dict[str, float]:
    """Temps de chargement (hors imports des bibliothèques) et RAM max, dans un processus neuf."""
    probe = (
        f"import time, resource, sys; sys.path.insert(0, {str(ROOT)!r}); "
        f"import numpy, pandas, scipy.sparse, requests; t = time.perf_counter(); {code}; "
        "print(time.perf_counter() - t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.split()
    rss = float(out[1]) / (1 << 20 if sys.platform == "darwin" else 1 << 10)  # octets (macOS) / Ko (Linux)
    return {"load_s": round(float(out[0]), 2), "max_rss_mb": round(rss)}


def main() -> None:
    legacy_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LEGACY
    with legacy_path.open("rb") as f:
        legacy = pickle.load(f)  # noqa: S301  fichier local produit par le projet
    old_df, mats = legacy["df"], legacy["matrices"]
    art = load_artifacts(ROOT / "artifacts")
    cat = art.catalog

    old_pos = {int(i): p for p, i in enumerate(old_df["id"])}
    new_to_old = np.array([old_pos[int(i)] for i in cat["id"]])
    old_to_new = {o: n for n, o in enumerate(new_to_old)}
    rng = np.random.default_rng(SEED)

    # 1. Accord signal par signal
    sample = rng.choice(art.n, 300, replace=False)
    signals = {}
    for s in SIGNALS:
        cors, overlaps = [], []
        for i in sample:
            if s == "collab" and not art.has_collab(int(i)):
                continue
            a = mats[s][new_to_old[i]][new_to_old].astype(np.float32)
            b = art.similarity(s, int(i))
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
            ta, tb = set(np.argsort(-a)[1:21]), set(np.argsort(-b)[1:21])
            overlaps.append(len(ta & tb) / 20)
        signals[s] = {"corr": round(float(np.mean(cors)), 3), "top20_overlap": round(float(np.mean(overlaps)), 3)}

    # 2. Jaccard des top-10 (ancien moteur vs nouveau moteur)
    films = rng.choice(art.n, N_FILMS, replace=False)
    jaccards, examples = [], []
    for i in films:
        old_top = {old_to_new.get(j, -1) for j in legacy_top(old_df, mats, int(new_to_old[i]), K)}
        new_top = {r.idx for r in recommend(art, int(i), k=K).items}
        jaccards.append(len(old_top & new_top) / len(old_top | new_top))
        if len(examples) < 5:
            examples.append(
                {
                    "film": cat.iloc[i]["title"],
                    "old": [cat.iloc[j]["title"] for j in old_top if j >= 0][:5],
                    "new": [cat.iloc[j]["title"] for j in new_top][:5],
                }
            )

    # 3. Chargement : avant / après
    before = measure(f"import pickle; pickle.load(open({str(legacy_path)!r}, 'rb'))")
    after = measure(
        "from pathlib import Path; from cinematch.data import load_artifacts; "
        f"load_artifacts(Path({str(ROOT / 'artifacts')!r}))"
    )
    sizes = {
        "legacy_mb": round(legacy_path.stat().st_size / 1e6, 1),
        "artifacts_mb": round(sum(f["bytes"] for f in art.manifest["files"].values()) / 1e6, 1),  # type: ignore[union-attr]
    }

    result = {
        "n_films_compared": N_FILMS,
        "k": K,
        "seed": SEED,
        "jaccard_top10_mean": round(float(np.mean(jaccards)), 3),
        "jaccard_top10_median": round(float(np.median(jaccards)), 3),
        "signals": signals,
        "load_before": before,
        "load_after": after,
        "size": sizes,
        "examples": examples,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "regression.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Non-régression : ancien pickle vs nouveaux artefacts",
        "",
        "Généré par `python scripts/compare_legacy.py`.",
        "",
        "## Taille et chargement",
        "",
        "| | Ancien (`recommender_data3.pkl`) | Nouveau (`artifacts/`) |",
        "| :-- | --: | --: |",
        f"| Taille | {sizes['legacy_mb']} Mo | {sizes['artifacts_mb']} Mo |",
        f"| Temps de chargement | {before['load_s']} s | {after['load_s']} s |",
        f"| RAM maximale du processus | {before['max_rss_mb']} Mo | {after['max_rss_mb']} Mo |",
        "",
        "## Accord par signal (300 films, lignes de similarité)",
        "",
        "| Signal | Corrélation | Recouvrement top-20 |",
        "| :-- | --: | --: |",
        *[f"| `{s}` | {v['corr']:.3f} | {v['top20_overlap']:.0%} |" for s, v in signals.items()],
        "",
        f"## Top-10 du moteur complet ({N_FILMS} films)",
        "",
        f"Jaccard moyen : **{result['jaccard_top10_mean']:.2f}** (médiane {result['jaccard_top10_median']:.2f}).",
        "",
        "Les écarts s'expliquent par :",
        "",
        "- `collab` est reconstruit à l'identique (cosinus item-item sur les notes brutes MovieLens) ;",
        "- `vibe` n'était pas versionné : il est reconstruit (accroche + genres + mots-clés, MiniLM), "
        "c'est la principale source d'écart ;",
        "- les métadonnées viennent de l'API TMDB actuelle (synopsis, mots-clés et popularité ont changé depuis l'export Kaggle de 2017) ;",
        "- les règles « langue étrangère » et « film populaire » sont maintenant actives ;",
        "- le cold start redistribue le poids réel de `collab` (et non une constante de 0,2) ;",
        "- 8 films supprimés de TMDB sont sortis du catalogue (6 195 films).",
    ]
    (REPORTS / "regression.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
