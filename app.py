import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Comparateur de textes touristiques",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# FUNCTIONS
# =========================
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def match_sentences(sentences1, sentences2):
    emb1 = model.encode(sentences1)
    emb2 = model.encode(sentences2)

    pairs = []

    for i, e1 in enumerate(emb1):
        sims = [cosine_similarity(e1, e2) for e2 in emb2]

        best_idx = int(np.argmax(sims)) if sims else None
        best_score = float(max(sims)) if sims else 0

        pairs.append({
            "text1": sentences1[i],
            "text2": sentences2[best_idx] if best_idx is not None else "",
            "score": best_score * 100
        })

    return pairs


def recommendation(global_score):

    if global_score > 80:
        return "🟢 Texte 2 très proche → aucune mise à jour nécessaire"
    elif global_score > 60:
        return "🟠 Texte 2 partiellement différent → légère optimisation recommandée"
    else:
        return "🔴 Texte 2 trop différent → mise à jour fortement recommandée"


# =========================
# UI
# =========================
st.title("🔍 Comparateur intelligent de textes touristiques")
st.write("Analyse sémantique + comparaison structurée + recommandation")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte 1 (référence)", height=250)

with col2:
    text2 = st.text_area("📄 Texte 2 (à comparer)", height=250)

# =========================
# ACTION
# =========================
if st.button("Comparer"):

    if text1 and text2:

        sentences1 = split_sentences(text1)
        sentences2 = split_sentences(text2)

        if not sentences1 or not sentences2:
            st.warning("Textes insuffisants pour analyse.")
            st.stop()

        pairs = match_sentences(sentences1, sentences2)

        global_score = np.mean([p["score"] for p in pairs])

        # =========================
        # 1. SCORE GLOBAL
        # =========================
        st.subheader("📊 Score global de similarité")
        st.metric("Similarité", f"{global_score:.2f}%")

        st.progress(min(global_score / 100, 1.0))

        # =========================
        # 3. RECOMMANDATION
        # =========================
        st.subheader("🧠 Recommandation")
        st.info(recommendation(global_score))

        # =========================
        # 2. TABLEAU COMPARATIF
        # =========================
        st.subheader("🔎 Comparatif détaillé")

        df = pd.DataFrame(pairs)

        # classification visuelle
        def label(score):
            if score > 75:
                return "🟢 similaire"
            elif score > 50:
                return "🟠 partiel"
            else:
                return "🔴 différent"

        df["niveau"] = df["score"].apply(label)
        df["score"] = df["score"].round(2)

        st.dataframe(
            df[["text1", "text2", "score", "niveau"]],
            use_container_width=True
        )

    else:
        st.warning("Veuillez remplir les deux textes.")
