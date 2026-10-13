"""Moteur de recommandation hybride de CineMatch (sans dépendance à Streamlit).

Le score final d'un film candidat est une somme pondérée de 8 matrices de
similarité pré-calculées (n_films x n_films). Les poids sont adaptés au film
source (genre, langue, popularité) par `smart_weights`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nom affiché (UI) -> clé de la matrice dans le pickle
SIGNAL_TO_MATRIX = {
    "Histoire": "sbert",
    "Sujets": "keywords",
    "Vibe": "vibe",
    "Genre": "genre",
    "Casting": "credits",
    "Origine": "origin",
    "Popularité": "context",
    "Public": "collab",
}

BASE_WEIGHTS = {
    "Histoire": 0.15, "Sujets": 0.25, "Vibe": 0.15,
    "Genre": 0.10, "Casting": 0.10, "Origine": 0.00,
    "Popularité": 0.05, "Public": 0.20,
}


def smart_weights(row: pd.Series) -> dict[str, float]:
    """Adapte les poids de base aux caractéristiques du film source.

    `original_language` et `vote_count` sont optionnels : s'ils sont absents du
    DataFrame (c'est le cas du pickle actuel), les règles associées restent
    inactives.
    """
    genres = set(row["genres_list"]) if isinstance(row["genres_list"], list) else set()
    lang = row.get("original_language", "en")
    vote_count = row.get("vote_count", 0)

    w = dict(BASE_WEIGHTS)

    if "Animation" in genres or any(ch.isdigit() for ch in row["title"]):
        w["Sujets"] += 0.15
        w["Public"] += 0.10
    if "Horror" in genres:
        w["Vibe"] += 0.20
        w["Sujets"] += 0.10
        w["Casting"] -= 0.05
    if lang != "en":
        w["Origine"] += 0.30
        w["Histoire"] -= 0.10
    if vote_count > 3000:
        w["Public"] += 0.15
        w["Popularité"] += 0.05
    return w


def recommend(
    df: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    idx: int,
    k: int = 4,
) -> tuple[list[dict], dict[str, float]]:
    """Retourne les `k` films les plus proches du film `idx` et les poids utilisés.

    `idx` est une position (0..len(df)-1) dans `df`, pas un titre : les titres
    ne sont pas uniques (ex. deux « Aladdin »).
    """
    weights = smart_weights(df.iloc[idx])

    # Cold start : aucun signal collaboratif -> on redistribue son poids
    if matrices["collab"][idx].sum() == 0:
        weights["Public"] = 0.0
        bonus = 0.2 / 3
        for name in ("Histoire", "Sujets", "Vibe"):
            weights[name] += bonus

    scores = np.zeros(len(df), dtype=np.float32)
    for name, w in weights.items():
        if w > 0:
            scores += w * matrices[SIGNAL_TO_MATRIX[name]][idx].astype(np.float32)

    # k+1 candidats car le film source se classe presque toujours en tête
    n_cand = min(k + 1, len(scores))
    top = np.argpartition(scores, -n_cand)[-n_cand:]
    top = top[np.argsort(scores[top])[::-1]]
    top = [int(i) for i in top if i != idx][:k]

    total_w = sum(weights.values())
    results = []
    for i in top:
        row = df.iloc[i]
        results.append({
            "id": row["id"],
            "title": row["title"],
            "year": int(row["year"]) if row["year"] > 0 else None,
            "director": row["director"] or "",
            "score": float(scores[i] / total_w) if total_w > 0 else 0.0,
        })
    return results, weights
