"""Évaluation hors-ligne complète : `python -m evaluation.run [--trials 200]`.

Produit dans `reports/` :
- `eval.json` : résultats bruts (métriques + IC bootstrap), graine, versions ;
- `eval.md`   : tableaux baselines vs hybride, ablation, segments ;
- `ablation.png` : perte de NDCG@10 quand on retire chaque signal ou règle ;
- `best_weights.json` : poids de base optimisés sur la validation.

Le test n'est utilisé qu'une fois, après l'optimisation des poids sur la validation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cinematch.config import RULES, SIGNALS, WeightConfig
from cinematch.data import Artifacts, load_artifacts
from evaluation import metrics as M
from evaluation.baselines import Method, hybrid, standard_methods, weight_vector
from evaluation.dataset import Query, make_eval_data
from pipeline.common import ARTIFACTS_DIR, ROOT, collab_factor, ensure_movielens

REPORTS = ROOT / "reports"
SEED = 42
KS = (4, 10)
SEGMENTS = {"standard": False, "cold_start": True}
SEGMENT_LABELS = {"standard": "Standard", "cold_start": "Cold start simulé"}
SIGNAL_LABELS = {
    "sbert": "Histoire (sbert)",
    "keywords": "Sujets (keywords)",
    "vibe": "Ambiance (vibe)",
    "genre": "Genre",
    "credits": "Casting (credits)",
    "origin": "Origine",
    "context": "Époque & popularité (context)",
    "collab": "Public (collab)",
}
METRIC_KEYS = ("ndcg@10", "hr@10", "mrr@10", "p@4", "p@10", "r@4", "r@10")


@dataclass
class Run:
    """Métriques par requête d'une méthode sur un segment."""

    per_query: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in METRIC_KEYS})
    users: list[int] = field(default_factory=list)
    lists: list[list[int]] = field(default_factory=list)


def top_k(scores: NDArray[np.float32], q: Query, k: int) -> list[int]:
    s = scores.astype(np.float32, copy=True)
    s[list(q.seen)] = -np.inf
    s[q.idx] = -np.inf
    cand = np.argpartition(s, -k)[-k:]
    return [int(i) for i in cand[np.argsort(s[cand])[::-1]]]


def record(run: Run, q: Query, ranked: list[int]) -> None:
    rel = q.relevant
    run.per_query["ndcg@10"].append(M.ndcg_at_k(ranked, rel, 10))
    run.per_query["hr@10"].append(M.hit_rate_at_k(ranked, rel, 10))
    run.per_query["mrr@10"].append(M.mrr_at_k(ranked, rel, 10))
    for k in KS:
        run.per_query[f"p@{k}"].append(M.precision_at_k(ranked, rel, k))
        run.per_query[f"r@{k}"].append(M.recall_at_k(ranked, rel, k))
    run.users.append(q.user)
    run.lists.append(ranked)


def evaluate(
    art: Artifacts, queries: Sequence[Query], methods: Sequence[Method], segments: dict[str, bool] = SEGMENTS
) -> dict[str, dict[str, Run]]:
    """Évalue toutes les méthodes en ne calculant qu'une fois les similarités de chaque requête."""
    runs = {m.name: {seg: Run() for seg in segments} for m in methods}
    for q in queries:
        S = art.similarities(q.idx)
        for m in methods:
            for seg, cold in segments.items():
                record(runs[m.name][seg], q, top_k(m.score(q, S, cold), q, 10))
    return runs


def summarize(run: Run, art: Artifacts, pop_pct: NDArray[np.float64], n_boot: int) -> dict[str, object]:
    users = np.array(run.users)
    out: dict[str, object] = {"n_queries": len(users), "n_users": len(np.unique(users))}
    for m in METRIC_KEYS:
        mean, lo, hi = M.bootstrap_ci(np.array(run.per_query[m]), users, n_boot=n_boot, seed=SEED)
        out[m] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["coverage@10"] = round(M.catalog_coverage(run.lists, art.n), 4)
    out["diversity@10"] = round(
        float(np.mean([M.intra_list_diversity(lst, art.embeddings["sbert"]) for lst in run.lists])), 4
    )
    out["popularity_pct@10"] = round(float(np.mean([pop_pct[lst].mean() for lst in run.lists])), 1)
    return out


def tune(art: Artifacts, queries: Sequence[Query], n_trials: int) -> tuple[WeightConfig, float, float]:
    """Optimise les 8 poids de base (règles conservées) sur la validation.

    Objectif : moyenne des NDCG@10 standard et cold start, car l'app sert les deux cas.
    """
    import optuna

    stacks = [art.similarities(q.idx) for q in queries]
    rows = [art.catalog.iloc[q.idx] for q in queries]

    def objective_for(cfg: WeightConfig) -> float:
        total = 0.0
        for cold in (False, True):
            vals = []
            for q, S, row in zip(queries, stacks, rows, strict=True):
                w = weight_vector(row, cfg, cold)
                vals.append(M.ndcg_at_k(top_k(w @ S / max(float(w.sum()), 1e-9), q, 10), q.relevant, 10))
            total += float(np.mean(vals))
        return total / 2

    def objective(trial: optuna.Trial) -> float:
        base = {s: trial.suggest_float(s, 0.0, 1.0) for s in SIGNALS}
        return objective_for(WeightConfig().with_base(base))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    default = WeightConfig()
    study.enqueue_trial(dict(default.base))  # point de départ : les poids manuels
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    norm = sum(best.values())
    tuned = default.with_base({s: round(best[s] / norm, 4) for s in SIGNALS})
    return tuned, objective_for(default), study.best_value


def ablation_chart(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r["label"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.6), sharey=True)
    for ax, seg in zip(axes, SEGMENTS, strict=True):
        vals = [float(r[seg]) for r in rows]  # type: ignore[arg-type]
        # Bleu : l'élément aide ; rouge : il pénalise ; gris : effet négligeable (< 0,05 point)
        colors = ["#b4b2ab" if abs(v) < 0.05 else "#2a78d6" if v < 0 else "#e34948" for v in vals]
        ax.barh(labels, vals, color=colors, height=0.6)
        ax.axvline(0, color="#52514e", linewidth=1)
        ax.set_title(f"{SEGMENT_LABELS[seg]}", fontsize=11, color="#0b0b0b", loc="left")
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=9, left=False)
        span = max(abs(v) for v in vals) or 1.0
        ax.set_xlim(min(min(vals), 0) - 0.22 * span, max(max(vals), 0) + 0.22 * span)
        ax.set_xlabel("Δ NDCG@10 (points) quand l'élément est retiré", fontsize=9, color="#52514e")
        for y, v in enumerate(vals):
            ax.annotate(
                f"{v:+.2f}",
                (v, y),
                xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if v >= 0 else "right",
                va="center",
                fontsize=8,
                color="#0b0b0b",
            )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Ablation : bleu = l'élément aide (sa suppression coûte), rouge = il pénalise, gris = négligeable",
        fontsize=10,
        color="#52514e",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fmt(stat: dict[str, object], key: str) -> str:
    v = stat[key]
    assert isinstance(v, dict)
    return f"{v['mean']:.3f} [{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--per-user", type=int, default=5)
    args = parser.parse_args()
    t0 = time.perf_counter()

    art = load_artifacts(ARTIFACTS_DIR)
    ratings = pd.read_csv(ensure_movielens() / "ratings.csv")
    data = make_eval_data(ratings, art.catalog["movielens_id"], args.val_frac, args.per_user, SEED)
    # Pas de fuite : le signal collaboratif ne voit que la partie train
    art = art.with_factor("collab", collab_factor(data.train, art.catalog["movielens_id"]))
    votes = art.catalog["vote_count"].to_numpy()
    pop_pct = pd.Series(votes).rank(pct=True).to_numpy() * 100
    print(f"Validation : {len(data.val_queries)} requêtes · test : {len(data.test_queries)} requêtes")

    tuned, val_manual, val_tuned = tune(art, data.val_queries, args.trials)
    (REPORTS / "best_weights.json").parent.mkdir(exist_ok=True)
    (REPORTS / "best_weights.json").write_text(json.dumps(tuned.to_dict(), indent=2) + "\n")
    print(f"Optuna ({args.trials} essais) : NDCG@10 validation {val_manual:.4f} -> {val_tuned:.4f}")

    # Test : une seule passe pour toutes les méthodes et l'ablation
    methods = standard_methods(art, data.train, tuned)
    abl_methods = [hybrid(art, f"-{s}", drop=frozenset({s})) for s in SIGNALS]
    abl_methods += [hybrid(art, f"-rule:{r}", cfg=WeightConfig().without_rules(r)) for r in RULES]
    runs = evaluate(art, data.test_queries, methods + abl_methods)

    results = {
        name: {seg: summarize(run, art, pop_pct, args.bootstrap) for seg, run in segs.items()}
        for name, segs in runs.items()
    }
    ref = "Hybride actuel (base + règles)"
    ablation = []
    for s in SIGNALS:
        ablation.append(
            {
                "label": SIGNAL_LABELS[s],
                "kind": "signal",
                "key": s,
                **{
                    seg: round(
                        100 * (results[f"-{s}"][seg]["ndcg@10"]["mean"] - results[ref][seg]["ndcg@10"]["mean"]), 2
                    )  # type: ignore[index]
                    for seg in SEGMENTS
                },
            }
        )
    for r in RULES:
        ablation.append(
            {
                "label": f"Règle « {r} »",
                "kind": "rule",
                "key": r,
                **{
                    seg: round(
                        100 * (results[f"-rule:{r}"][seg]["ndcg@10"]["mean"] - results[ref][seg]["ndcg@10"]["mean"]), 2
                    )  # type: ignore[index]
                    for seg in SEGMENTS
                },
            }
        )
    REPORTS.mkdir(exist_ok=True)
    ablation_chart(ablation, REPORTS / "ablation.png")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "protocol": {
            "dataset": "MovieLens ml-latest-small, restreint au catalogue",
            "split": "temporel par utilisateur, 80 % train / 20 % test",
            "positive": "note >= 4",
            "query": f"jusqu'à {args.per_user} films aimés les plus récents du train par utilisateur",
            "relevant": "films aimés dans la partie test",
            "excluded_from_ranking": "film requête + films déjà notés dans le train",
            "collab": "recalculé sur la partie train uniquement",
            "users_split": f"{args.val_frac:.0%} validation (poids), reste test (une seule passe)",
            "cold_start": "simulé : signal collaboratif du film requête masqué, poids redistribué",
            "bootstrap": f"{args.bootstrap} tirages sur les utilisateurs, IC 95 %",
            "n_val_queries": len(data.val_queries),
            "n_test_queries": len(data.test_queries),
            "n_test_users": len({q.user for q in data.test_queries}),
            "catalog_size": art.n,
            "catalog_cold_share": round(1 - art.collab_coverage, 4),
        },
        "tuning": {
            "trials": args.trials,
            "objective": "moyenne NDCG@10 standard et cold start (validation)",
            "val_manual": round(val_manual, 4),
            "val_tuned": round(val_tuned, 4),
            "best_base": tuned.base,
        },
        "methods": {m.name: results[m.name] for m in methods},
        "ablation": ablation,
        "versions": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("numpy", "pandas", "scipy", "optuna")},
        },
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    (REPORTS / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (REPORTS / "eval.md").write_text(render_markdown(report))
    print((REPORTS / "eval.md").read_text())


def render_markdown(rep: dict) -> str:  # type: ignore[type-arg]
    p = rep["protocol"]
    lines = [
        "# CineMatch : évaluation hors-ligne",
        "",
        f"Généré par `python -m evaluation.run` le {rep['generated_at']} (graine {rep['seed']}). "
        "Résultats bruts : [`eval.json`](eval.json).",
        "",
        "## Protocole",
        "",
        *[f"- **{k.replace('_', ' ')}** : {v}" for k, v in p.items()],
        "",
        "Le *cold start simulé* rejoue chaque requête en masquant le signal collaboratif du film "
        "requête, comme pour les 43 % du catalogue sans note MovieLens.",
        "",
    ]
    for seg in SEGMENTS:
        lines += [
            f"## Baselines vs hybride ({SEGMENT_LABELS[seg].lower()})",
            "",
            "Moyenne [IC 95 % bootstrap].",
            "",
            "| Méthode | NDCG@10 | HitRate@10 | MRR@10 | P@4 | R@10 | Couverture | Diversité | Popularité (centile) |",
            "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
        ]
        for name, segs in rep["methods"].items():
            st = segs[seg]
            lines.append(
                f"| {name} | {fmt(st, 'ndcg@10')} | {fmt(st, 'hr@10')} | {st['mrr@10']['mean']:.3f} | "
                f"{st['p@4']['mean']:.3f} | {st['r@10']['mean']:.3f} | {st['coverage@10']:.1%} | "
                f"{st['diversity@10']:.2f} | {st['popularity_pct@10']:.0f} |"
            )
        lines.append("")
    t = rep["tuning"]
    lines += [
        "## Optimisation des poids (Optuna)",
        "",
        f"{t['trials']} essais TPE sur la validation, objectif : {t['objective']}. "
        f"Validation : {t['val_manual']:.4f} (poids manuels) → {t['val_tuned']:.4f} (poids optimisés).",
        "",
        "| Signal | Poids manuel | Poids optimisé |",
        "| :-- | --: | --: |",
        *[f"| {SIGNAL_LABELS[s]} | {WeightConfig().base[s]:.2f} | {t['best_base'][s]:.2f} |" for s in SIGNALS],
        "",
        "## Ablation (test)",
        "",
        "Variation de NDCG@10, en points, quand on retire un signal ou une règle de l'hybride actuel. "
        "Négatif : l'élément aide.",
        "",
        "![Ablation](ablation.png)",
        "",
        "| Élément retiré | Standard | Cold start simulé |",
        "| :-- | --: | --: |",
        *[f"| {a['label']} | {a['standard']:+.2f} | {a['cold_start']:+.2f} |" for a in rep["ablation"]],
        "",
        "## Lecture",
        "",
        "- *Couverture* : part du catalogue apparaissant dans au moins un top-10.",
        "- *Diversité* : 1 − similarité SBERT moyenne entre les films d'une même liste.",
        "- *Popularité* : centile moyen (nombre de votes TMDB) des films recommandés ; 50 = catalogue moyen.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
