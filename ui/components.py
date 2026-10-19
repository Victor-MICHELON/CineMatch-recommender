"""Composants Streamlit de CineMatch : styles, radar, film source, cartes, panneaux."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cinematch.config import SIGNALS, WeightConfig
from cinematch.data import Artifacts
from cinematch.engine import Recommendation, RecommendationResult
from cinematch.tmdb import MovieMedia, page_url
from ui.i18n import REASONS, RULE_LABELS, SIGNAL_LABELS, TEXTS

UI_DIR = Path(__file__).parent
ASSETS_DIR = UI_DIR.parent / "assets"
NO_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='750'>"
    "<rect width='100%' height='100%' fill='%23161b22'/>"
    "<text x='50%' y='50%' fill='%23aaa' font-family='sans-serif' font-size='28' "
    "text-anchor='middle'>No image</text></svg>"
)


def tmdb_key() -> str | None:
    """Clé TMDB : variable d'environnement, sinon secrets Streamlit."""
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("TMDB_API_KEY")
    except Exception:  # pas de fichier secrets.toml
        return None


def load_css() -> None:
    st.markdown(f"<style>{(UI_DIR / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def header() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        banner = ASSETS_DIR / "Banner_CineMatch.png"
        if banner.exists():
            st.image(str(banner), width="stretch")
        else:
            st.markdown("<h1 style='text-align:center'>🎬 CineMatch AI</h1>", unsafe_allow_html=True)


def radar(weights: dict[str, float], lang: str) -> go.Figure:
    labels = [SIGNAL_LABELS[lang][s] for s in SIGNALS]
    values = [weights[s] for s in SIGNALS]
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line={"color": "#FF6B00", "width": 3},
            fillcolor="rgba(255, 107, 0, 0.25)",
            opacity=0.9,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    grid = "rgba(255,255,255,0.15)"
    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, max(max(values), 0.01) * 1.15],
                "gridcolor": grid,
                "tickfont": {"size": 10},
            },
            "angularaxis": {"gridcolor": grid, "tickfont": {"size": 10}},
            "bgcolor": "rgba(18, 18, 18, 0.6)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=320,
        font={"color": "white", "size": 11, "family": "Inter"},
        showlegend=False,
    )
    return fig


def sidebar_analysis(result: RecommendationResult, lang: str) -> None:
    t = TEXTS[lang]
    st.plotly_chart(radar(result.weights, lang), width="stretch")
    if result.rules:
        st.caption(f"{t['rules']} : " + ", ".join(RULE_LABELS[lang][r] for r in result.rules))
    if result.cold_start:
        st.caption("❄️ " + t["cold_start"])


def source_film(art: Artifacts, queries: list[int], media: dict[int, MovieMedia | None], lang: str) -> None:
    t = TEXTS[lang]
    row = art.catalog.iloc[queries[0]]
    with st.container(border=True):
        st.markdown(f"### 📡 {t['based_on']} : **{html.escape(str(row['title']))}**")
        poster_col, details = st.columns([1, 3])
        with poster_col:
            m = media.get(int(row["id"]))
            st.image(m.poster_url if m and m.poster_url else NO_POSTER, width="stretch")
        with details:
            parts = []
            if row["year"] > 0:
                parts.append(f"**🎬 {int(row['year'])}**")
            if row["director"]:
                parts.append(f"🎥 _{row['director']}_")
            if row["genres_list"]:
                parts.append(" · ".join(row["genres_list"][:3]))
            st.markdown(" • ".join(parts))
            overview = str(row["overview"] or "")
            overview = overview[:320] + "…" if len(overview) > 320 else overview
            st.markdown(
                f"<p class='cm-overview'>{html.escape(overview) or t['no_overview']}</p>",
                unsafe_allow_html=True,
            )
            if len(queries) > 1:
                others = ", ".join(html.escape(str(art.catalog.iloc[q]["title"])) for q in queries[1:])
                st.markdown(f"<p class='cm-chips'>{t['also']} : {others}</p>", unsafe_allow_html=True)


def reason_text(item: Recommendation, art: Artifacts, queries: list[int], lang: str) -> str:
    """Les deux signaux dominants, en langage courant (« même réalisateur · ambiance proche »)."""
    directors = {art.catalog.iloc[q]["director"] for q in queries} - {""}
    reasons = []
    for s in item.top_signals(2):
        key = "same_director" if s == "credits" and item.director in directors else s
        reasons.append(REASONS[lang][key])
    return " · ".join(reasons)


def movie_card(item: Recommendation, media: MovieMedia | None, why: str, lang: str) -> str:
    t = TEXTS[lang]
    title = html.escape(item.title)
    director = html.escape(item.director)
    score = min(int(item.score * 100), 99)
    poster = media.poster_url if media and media.poster_url else NO_POSTER
    links = [f"<a href='{page_url(item.id)}' target='_blank' rel='noopener'>{t['tmdb']} ↗</a>"]
    if media and media.trailer_url:
        links.append(f"<a href='{html.escape(media.trailer_url)}' target='_blank' rel='noopener'>▶ {t['trailer']}</a>")
    return f"""
<div class="cm-card">
  <div class="cm-poster">
    <img src="{poster}" alt="{t["tmdb"]} : {title}" loading="lazy">
    <span class="cm-badge">{score}% {t["match"]}</span>
  </div>
  <div class="cm-body">
    <div class="cm-title" title="{title}">{title}</div>
    <div class="cm-meta"><span>📅 {item.year or "—"}</span><span title="{director}">🎥 {director}</span></div>
    <div class="cm-why">{t["why"]} : <b>{html.escape(why)}</b></div>
    <div class="cm-bar" role="img" aria-label="{score}%"><div style="width:{score}%"></div></div>
    <div class="cm-links">{"".join(links)}</div>
  </div>
</div>"""


def results_grid(
    items: list[Recommendation],
    art: Artifacts,
    queries: list[int],
    media: dict[int, MovieMedia | None],
    lang: str,
) -> None:
    for start in range(0, len(items), 4):
        cols = st.columns(4, gap="medium")
        for col, item in zip(cols, items[start : start + 4], strict=False):
            with col:
                why = reason_text(item, art, queries, lang)
                st.markdown(movie_card(item, media.get(item.id), why, lang), unsafe_allow_html=True)


def empty_state(lang: str) -> None:
    t = TEXTS[lang]
    st.markdown(
        f"<div class='cm-empty'><h2>{t['empty_title']}</h2><p>{t['empty_body']}</p>"
        f"<p style='opacity:.75;font-size:.9rem'>{t['empty_hint']}</p></div>",
        unsafe_allow_html=True,
    )


def filters_panel(art: Artifacts, lang: str) -> np.ndarray | None:
    """Filtres décennie / genre / langue / durée. Renvoie le masque des films autorisés."""
    t = TEXTS[lang]
    cat = art.catalog
    with st.expander(t["filters"]):
        c1, c2, c3 = st.columns(3)
        decades = sorted({int(y) // 10 * 10 for y in cat["year"] if y > 0})
        dec = c1.multiselect(t["decades"], decades, format_func=lambda d: f"{d}s", key="f_decades")
        genres = sorted({g for gs in cat["genres_list"] for g in gs})
        gen = c2.multiselect(t["genres"], genres, key="f_genres")
        langs = cat["original_language"].value_counts().index.tolist()
        lng = c3.multiselect(t["languages"], langs, key="f_langs")
        rt = st.slider(t["runtime"], 0, 240, (0, 240), step=10, key="f_runtime")
    mask = np.ones(art.n, dtype=bool)
    if dec:
        mask &= (cat["year"] // 10 * 10).isin(dec).to_numpy()
    if gen:
        mask &= cat["genres_list"].map(lambda gs: bool(set(gs) & set(gen))).to_numpy()
    if lng:
        mask &= cat["original_language"].isin(lng).to_numpy()
    if rt != (0, 240):
        runtime = cat["runtime"].to_numpy()
        mask &= (runtime >= rt[0]) & ((runtime <= rt[1]) | (rt[1] == 240))
    return None if mask.all() else mask


def advanced_panel(lang: str, profiles: dict[str, WeightConfig]) -> dict[str, Any]:
    """Profil de poids (règles manuelles ou optimisées), curseurs libres et diversité."""
    t = TEXTS[lang]
    labels = SIGNAL_LABELS[lang]
    weights: dict[str, float] | None = None
    with st.expander(t["advanced"]):
        name = st.radio(
            "Profile",
            list(profiles),
            format_func=lambda p: t[f"profile_{p}"],
            horizontal=True,
            label_visibility="collapsed",
            key="profile",
        )
        cfg = profiles[name]
        if st.toggle(t["manual_weights"], key="manual"):
            if st.button(t["reset"], key="reset"):
                for s in SIGNALS:
                    st.session_state.pop(f"w_{s}", None)
            cols = st.columns(4)
            weights = {
                s: cols[k % 4].slider(labels[s], 0.0, 1.0, float(cfg.base[s]), 0.05, key=f"w_{s}")
                for k, s in enumerate(SIGNALS)
            }
        diversity = st.slider(t["diversity"], 0.0, 1.0, 0.0, 0.1, help=t["diversity_help"], key="diversity")
    return {"cfg": cfg, "weights": weights, "diversity": diversity}


def pager(page: int, has_next: bool, lang: str) -> int | None:
    """Boutons précédent / suivant. Renvoie la nouvelle page si l'un est cliqué."""
    t = TEXTS[lang]
    prev_col, info_col, next_col = st.columns([1, 2, 1])
    info_col.markdown(f"<p style='text-align:center'>{t['page']} {page + 1}</p>", unsafe_allow_html=True)
    if prev_col.button(t["prev"], disabled=page == 0, width="stretch"):
        return page - 1
    if next_col.button(t["next"], disabled=not has_next, width="stretch"):
        return page + 1
    return None
