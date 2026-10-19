"""CineMatch : point d'entrée Streamlit (`streamlit run app.py`)."""

import os
from pathlib import Path

import streamlit as st

from cinematch import Artifacts, WeightConfig, load_artifacts, recommend
from cinematch.data import ensure_artifacts
from cinematch.tmdb import MediaCache, TMDBClient
from ui import components as ui
from ui.i18n import LANGS, TEXTS, TMDB_LANG

ROOT = Path(__file__).parent
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
ui.load_css()


@st.cache_resource(show_spinner=False)
def load() -> Artifacts:  # ~12 Mo d'artefacts, téléchargés au premier lancement si absents
    return load_artifacts(ensure_artifacts(Path(os.environ.get("CINEMATCH_ARTIFACTS_DIR", ROOT / "artifacts"))))


@st.cache_resource(show_spinner=False)
def media_cache() -> MediaCache:
    key = ui.tmdb_key()
    return MediaCache(TMDBClient(key) if key else None)


with st.sidebar:
    lang = st.radio(
        "Language", list(LANGS), format_func=LANGS.get, horizontal=True, label_visibility="collapsed", key="lang"
    )
t = TEXTS[lang]
try:
    with st.spinner(t["loading"]):
        art = load()
except Exception as exc:  # artefacts absents, corrompus, téléchargement impossible…
    st.error(f"{t['load_error']}\n\n`{type(exc).__name__}: {exc}`")
    st.stop()

tuned_path = ROOT / "reports" / "best_weights.json"
profiles = {"rules": WeightConfig()}
if tuned_path.exists():
    profiles["tuned"] = WeightConfig.from_json(tuned_path)

with st.sidebar:
    icon = ui.ASSETS_DIR / "Icon_CineMatch.png"
    if icon.exists():
        st.image(str(icon), width=120)
    st.title(t["sidebar_title"])
    analysis = st.container()
    st.info("💡 " + t["radar_help"])

ui.header()
labels = [f"{r.title} ({r.year})" if r.year > 0 else r.title for r in art.catalog.itertuples()]
col_search, col_btn = st.columns([4, 1], vertical_alignment="bottom")
queries = col_search.multiselect(
    t["search"],
    range(art.n),
    format_func=labels.__getitem__,
    max_selections=5,
    placeholder=t["search_placeholder"],
    label_visibility="collapsed",
    key="queries",
)
if col_btn.button(t["launch"], width="stretch"):
    st.session_state.launched, st.session_state.page = True, 0

mask = ui.filters_panel(art, lang)
opts = ui.advanced_panel(lang, profiles)
per_page = st.segmented_control(t["n_results"], [4, 8, 12], default=4, key="per_page") or 4

if not (queries and st.session_state.get("launched")):
    ui.empty_state(lang)
    st.stop()

signature = (tuple(queries), per_page, repr(opts), None if mask is None else mask.tobytes())
if st.session_state.get("signature") != signature:  # nouvelle recherche : retour à la page 1
    st.session_state.signature, st.session_state.page = signature, 0
page = st.session_state.page
result = recommend(
    art,
    queries,
    k=per_page * (page + 1),
    cfg=opts["cfg"],
    weights=opts["weights"],
    diversity=opts["diversity"],
    mask=mask,
)
items = result.items[per_page * page :]
ids = [int(art.catalog.iloc[q]["id"]) for q in queries[:1]] + [i.id for i in items]
media = media_cache().get_many(ids, TMDB_LANG[lang])

with analysis:
    ui.sidebar_analysis(result, lang)
ui.source_film(art, list(queries), media, lang)
st.markdown("---")
st.markdown(f"### {t['recos']}")
if not items:
    st.warning(t["no_results"])
else:
    ui.results_grid(items, art, list(queries), media, lang)
    if media_cache().client is None or not any(media.values()):
        st.caption(t["posters_off"])

new_page = ui.pager(page, has_next=len(items) == per_page, lang=lang)
if new_page is not None:
    st.session_state.page = new_page
    st.rerun()
