# 🎬 CineMatch AI  
### *Next-Gen Hybrid Movie Recommendation Engine*

> 🔗 **App Demo**: https://cinematch-reco.streamlit.app/

---

<p align="center">
  <img src="Data/screenshot1_app.png" alt="Screenshot 1" width="40%">
  <img src="Data/screenshot2_app.png" alt="Screenshot 2" width="40%">
  <img src="Data/screenshot3_app.png" alt="Screenshot 3" width="40%">
</p>


## 📌 Overview

**CineMatch AI** is a next-generation movie recommendation system designed to overcome the limitations of traditional keyword-based filtering.

🎯 **Goal:** Deliver recommendations that genuinely capture the **mood, tone, and vibe** of films through a **hybrid architecture** combining:

- **Advanced NLP (SBERT embeddings)** for semantic understanding of plot summaries  
- **Content-based filtering** using structured movie metadata  
- **Weighted vector similarity** for consistent and meaningful recommendations

---

##  Key Features

###  1. Hybrid Recommendation Engine
An architecture powered by three complementary signals:

- **Content-Based Filtering:**  
  Uses genres, cast, directors, and keywords from TMDB.

- **Collaborative / Popularity-Aware Filtering:**  
  Integrates global popularity metrics or user-inspired weighting.

- **Vector Similarity Layer:**  
  Efficient cosine similarity for instant recommendation retrieval.

---

###  2. State-of-the-Art NLP with SBERT
Movie synopses are encoded using **Sentence-BERT (All-MiniLM-L6-v2)**:

- Understands **semantic meaning**, not just shared keywords  
- Detects implicit vibes and atmospheres  
  *(e.g., "dystopian", "intimate", "psychological" even if not explicitly mentioned)*

→ Result: **More accurate, human-like recommendations**

---

###  3. UI/UX with Streamlit
A polished user interface featuring:

- **Glassmorphism** and custom CSS animations  
- Interactive **Radar Chart** built with Plotly  
- Clean, responsive and intuitive user experience

---

## System Architecture

The core system is built upon a **weighted similarity matrix** that integrates three families of embeddings:

### 1. SBERT Embeddings  
→ Capture the semantic meaning of movie synopses.

### 2. Metadata Embeddings  
→ Genres, cast, directors, keywords.

### 3. Popularity / Collaborative Signals  
→ Global popularity scores or user-based weighting mechanisms.

---

## Recommendation Pipeline

1. **Extract and encode** the selected movie  
2. **Apply dynamic weighting** to each embedding family  
3. **Compute vector distances**  
4. **Retrieve the top 4 closest movies**  
5. **Visualize** the recommendation rationale using Plotly  

---

## 🛠️ Tech Stack

**Language:**  
- Python 3.x

**NLP:**  
- Sentence-Transformers (SBERT)  
- HuggingFace ecosystem

**Machine Learning & Similarity:**  
- Scikit-learn  
- Numpy  

**Data Processing:**  
- Pandas  

**Frontend / UI:**  
- Streamlit  
- Custom CSS  

**Visualization:**  
- Plotly (Radar Chart)

**External APIs:**  
- TMDB API for movie posters and metadata retrieval

## 👤 Author

**Victor MICHELON**
🔗 [*Portfolio*  ](https://github.com/Victor-MICHELON/Victor_MICHELON_Portfolio.github.io)

 
