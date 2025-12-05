import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import plotly.graph_objects as go
import os
from functools import lru_cache
import time

# ==============================================================================
# 0. CONFIGURATION & DESIGN
# ==============================================================================
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Streamlit
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Reset et Base */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Fond animé avec particules */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0f 50%, #0a0a1a 100%) !important;
        color: #e5e5e5 !important;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(255, 107, 0, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(41, 98, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: breathe 8s ease-in-out infinite;
    }
    
    @keyframes breathe {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.7; }
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 17, 21, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 107, 0, 0.15) !important;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: linear-gradient(180deg, rgba(255, 107, 0, 0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ccc !important;
    }

    /* Selectbox Premium */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(31, 31, 31, 0.8) !important;
        backdrop-filter: blur(10px);
        color: white !important;
        border: 1px solid rgba(255, 107, 0, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(255, 107, 0, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 107, 0, 0.2) !important;
        transform: translateY(-2px);
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background: rgba(20, 20, 20, 0.98) !important;
        backdrop-filter: blur(15px);
    }
    
    [data-baseweb="select"] ul {
        background: rgba(20, 20, 20, 0.98) !important;
    }
    
    [data-baseweb="select"] li {
        background: transparent !important;
        color: white !important;
    }
    
    [data-baseweb="select"] li:hover {
        background: rgba(255, 107, 0, 0.2) !important;
    }

    /* Bouton Lancer Premium */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #E65100 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 14px 28px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 0, 0.4) !important;
        width: 100% !important;
        height: 56px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 10px 30px rgba(255, 107, 0, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01) !important;
    }

    /* Titres avec gradient animé */
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FF6B00, #ffffff, #2962FF);
        background-size: 200% 200%;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: gradientShift 5s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2, h3 {
        color: #fff !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 8px rgba(255, 107, 0, 0.3);
    }
    
    /* Divider stylisé */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 0, 0.4), transparent) !important;
        margin: 30px 0 !important;
    }
    
    /* Conteneur principal */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Spinner */
    [data-testid="stSpinner"] > div {
        border-top-color: #FF6B00 !important;
    }
    
    /* Images */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Markdown containers */
    [data-testid="stMarkdownContainer"] {
        color: #e5e5e5 !important;
    }
    
    /* Info box */
    [data-testid="stAlert"] {
        background: rgba(41, 98, 255, 0.15) !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid #2962FF !important;
        border-radius: 8px;
        color: #ccc !important;
    }
    
    /* Columns gap */
    [data-testid="column"] {
        padding: 0 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CHARGEMENT OPTIMISÉ
# ==============================================================================
@st.cache_data
def load_data():
    try:
        with open('data/recommender_data3.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['df'], data['indices'], data['matrices']
    except FileNotFoundError:
        return None, None, None

with st.spinner("🎬 Chargement de l'intelligence artificielle..."):
    df, indices, matrices = load_data()

if df is None: st.stop()

# ==============================================================================
# 2. MOTEUR INTELLIGENT (CORE)
# ==============================================================================

def get_smart_weights(movie_idx):
    """Calcule les poids idéaux sans ralentir l'app."""
    row = df.iloc[movie_idx]
    
    genres = set(row['genres_list']) if isinstance(row['genres_list'], list) else set()
    lang = row.get('original_language', 'en')
    vote_count = row.get('vote_count', 0)
    
    # Base
    w = {
        'Histoire': 0.15, 'Sujets': 0.25, 'Vibe': 0.15,
        'Genre': 0.10, 'Casting': 0.10, 'Origine': 0.00,
        'Popularité': 0.05, 'Public': 0.20
    }
    
    # Adaptation Rapide
    if 'Animation' in genres or any(char.isdigit() for char in row['title']):
        w['Sujets'] += 0.15; w['Public'] += 0.10
    if 'Horror' in genres:
        w['Vibe'] += 0.20; w['Sujets'] += 0.10; w['Casting'] -= 0.05
    if lang != 'en':
        w['Origine'] += 0.30; w['Histoire'] -= 0.10
    if vote_count > 3000:
        w['Public'] += 0.15; w['Popularité'] += 0.05
        
    return w

def get_recommendations(title):
    idx = indices.get(title)
    if idx is None: return [], {}
    if isinstance(idx, pd.Series): idx = idx.iloc[0]

    # 1. Poids Intelligents
    weights = get_smart_weights(idx)
    
    # Mapping interne
    key_map = {
        'Histoire': 'sbert', 'Sujets': 'keywords', 'Vibe': 'vibe',
        'Genre': 'genre', 'Casting': 'credits', 'Origine': 'origin',
        'Popularité': 'context', 'Public': 'collab'
    }

    # 2. Cold Start Check
    if matrices['collab'][idx].sum() == 0:
        weights['Public'] = 0
        bonus = 0.2 / 3
        weights['Histoire'] += bonus; weights['Sujets'] += bonus; weights['Vibe'] += bonus

    # 3. Calcul Vectoriel Rapide
    scores = np.zeros(len(df))
    for ui_key, val in weights.items():
        if val > 0:
            scores += val * matrices[key_map[ui_key]][idx]

    # 4. Tri Optimisé
    top_indices = np.argpartition(scores, -5)[-5:]
    top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
    top_indices = [i for i in top_indices if i != idx][:4]
    
    results = []
    total_w = sum(weights.values())
    
    for i in top_indices:
        row = df.iloc[i]
        norm_score = scores[i] / total_w if total_w > 0 else 0
        results.append({
            'id': row['id'], 'title': row['title'],
            'year': int(row['year']) if row['year'] > 0 else '',
            'director': row['director'], 'score': norm_score
        })
        
    return results, weights

# ==============================================================================
# 3. API IMAGE (CACHÉE)
# ==============================================================================
@lru_cache(maxsize=1024)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=f3b44d2ba65ebeee22e40bac40129d40"
    try:
        data = requests.get(url, timeout=0.6).json()
        if data.get('poster_path'): return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except: pass
    return "https://via.placeholder.com/500x750/161b22/FFFFFF?text=No+Image"

# ==============================================================================
# 4. INTERFACE UTILISATEUR
# ==============================================================================

# --- SIDEBAR (Radar Chart) ---
with st.sidebar:
    icon_path = os.path.join("Data", "Icon_CineMatch.png")
    
    if os.path.exists(icon_path):
        st.image(icon_path, width=120)
    else:
        st.markdown("### 🎬 CineMatch")
        
    st.title("Analyse IA")
    st.markdown("---")
    
    chart_placeholder = st.empty()
    st.info("💡 Le graphique radar montre comment le système pondère les critères pour optimiser vos recommandations.")

# --- HEADER AVEC BANNIERE ---
banner_path = os.path.join("data", "banner_CineMatch.png")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    else:
        st.markdown("<h1 style='text-align: center;'>🎬 CineMatch AI</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialisation de l'état
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False

# Barre de recherche + Bouton
col_search, col_btn = st.columns([4, 1])

with col_search:
    selected_movie = st.selectbox(
        "Rechercher un film...",
        options=df['title'].values,
        index=None,
        placeholder="🔍 Tapez un titre (ex: Interstellar, Parasite, The Matrix...)",
        label_visibility="collapsed"
    )

with col_btn:
    if st.button("🚀 Lancer", use_container_width=True):
        st.session_state.search_triggered = True

# --- LOGIQUE D'AFFICHAGE ---
if selected_movie and st.session_state.search_triggered:
    
    with st.spinner("🎯 Analyse vectorielle en cours..."):
        recos, active_weights = get_recommendations(selected_movie)
        src_row = df[df['title'] == selected_movie].iloc[0]

    # --- RADAR CHART (SIDEBAR) ---
    with chart_placeholder:
        categories = list(active_weights.keys())
        values = list(active_weights.values())
        fig = go.Figure(data=go.Scatterpolar(
            r=values, 
            theta=categories, 
            fill='toself', 
            line=dict(color='#FF6B00', width=3),
            fillcolor='rgba(255, 107, 0, 0.25)',
            opacity=0.9
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, max(values)*1.15], 
                    gridcolor='rgba(255,255,255,0.15)',
                    tickfont=dict(size=10)
                ),
                bgcolor='rgba(18, 18, 18, 0.6)',
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.15)',
                    tickfont=dict(size=10)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=10, r=10, t=20, b=20),
            height=320, 
            font=dict(color='white', size=11, family='Inter'), 
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- HERO SECTION (Film Source) avec style inline ---
    st.markdown("""
    <div style='
        background: rgba(20, 20, 20, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin: 20px 0;
        border: 1px solid rgba(255, 107, 0, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    '>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### 📡 Analyse basée sur : **{src_row['title']}**")
    
    col_poster, col_details = st.columns([1, 3])
    with col_poster:
        st.markdown("""
        <div style='
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        '>
        """, unsafe_allow_html=True)
        st.image(fetch_poster(src_row['id']), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_details:
        st.markdown(f"**🎬 {int(src_row['year'])}** • 🎥 _{src_row['director']}_")
        st.markdown("<br>", unsafe_allow_html=True)
        overview_text = src_row['overview'][:320] + "..." if len(src_row['overview']) > 320 else src_row['overview']
        st.markdown(f"<p style='color: #ccc; line-height: 1.7; font-size: 0.95rem;'>{overview_text}</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- RECOMMANDATIONS (GRID) ---
    st.markdown("### ✨ Recommandations sur mesure")
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(4, gap="medium")
    for idx, col in enumerate(cols):
        if idx < len(recos):
            movie = recos[idx]
            with col:
                poster = fetch_poster(movie['id'])
                score = int(movie['score'] * 100)
                if score > 99: score = 99
                
                # Cartes avec HTML inline pour compatibilité totale
                st.markdown(f"""
                <div style='
                    background: rgba(24, 24, 24, 0.8);
                    backdrop-filter: blur(15px);
                    border-radius: 16px;
                    overflow: hidden;
                    transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                    border: 1px solid rgba(51, 51, 51, 0.6);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
                    cursor: pointer;
                '>
                    <div style='position:relative; overflow:hidden;'>
                        <img src="{poster}" style="width:100%; aspect-ratio: 2/3; object-fit: cover; transition: transform 0.5s ease;">
                        <div style="position:absolute; top:12px; right:12px; z-index:10;">
                            <span style='
                                background: linear-gradient(135deg, #FF6B00, #E65100);
                                color: white;
                                padding: 6px 12px;
                                border-radius: 8px;
                                font-size: 0.8rem;
                                font-weight: 800;
                                letter-spacing: 0.5px;
                                box-shadow: 0 4px 12px rgba(255, 107, 0, 0.6);
                                display: inline-block;
                            '>{score}% MATCH</span>
                        </div>
                    </div>
                    <div style='padding: 16px;'>
                        <div style='
                            font-weight: 700;
                            font-size: 1.1rem;
                            margin-bottom: 8px;
                            white-space: nowrap; 
                            overflow: hidden;
                            text-overflow: ellipsis;
                            color: #fff;
                        ' title="{movie['title']}">{movie['title']}</div>
                        <div style='
                            font-size: 0.85rem;
                            color: #999;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 14px;
                            font-weight: 500;
                        '>
                            <span>📅 {movie['year']}</span>
                            <span style="font-size:0.8rem; opacity:0.9">🎥 {movie['director'][:15]}..</span>
                        </div>
                        <div style='
                            width: 100%;
                            background: rgba(51, 51, 51, 0.6);
                            height: 6px;
                            border-radius: 3px;
                            overflow: hidden;
                        '>
                            <div style='
                                width: {score}%;
                                height: 100%;
                                background: linear-gradient(90deg, #FF6B00, #FF8F00);
                                border-radius: 3px;
                                box-shadow: 0 0 10px rgba(255, 107, 0, 0.6);
                                transition: width 1.5s ease-out;
                            '></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # Empty State Ultra Stylisé
    st.markdown("""
    <div style='
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 400px;
        background: rgba(20, 20, 20, 0.5);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 2px dashed rgba(255, 107, 0, 0.3);
        margin: 40px 0;
    '>
        <h2 style='color: #666; font-size: 2rem; margin-bottom: 10px;'>🎬 Prêt à découvrir ?</h2>
        <p style='color: #888; font-size: 1.1rem;'>Sélectionnez un film et cliquez sur <b style='color:#FF6B00;'>🚀 Lancer</b></p>
        <p style='margin-top:10px; font-size:0.9rem; opacity:0.6; color:#888;'>Notre IA analysera instantanément ses caractéristiques</p>
    </div>
    """, unsafe_allow_html=True)