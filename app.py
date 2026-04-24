import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Comparateur fournisseur / catalogue",
    page_icon="🔍",
    layout="wide"
)

# =========================
# MODEL NLP
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# FUNCTIONS
# =========================
def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_texts(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if len(s1) == 0 or len(s2) == 0:
        return 0, []

    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    results = []

    for i, e1 in enumerate(emb1):
        sims = [cosine(e1, e2) for e2 in emb2]

        best_score = max(sims)

        results.append({
            "Texte fournisseur (1)": s1[i],
            "Similarité (%)": round(best_score * 100, 2)
        })

    global_score = np.mean([r["Similarité (%)"] for r in results])

    return global_score, results


def recommendation(score):

    if score >= 85:
        return "🟢 Catalogue aligné avec le fournisseur (OK)"
    elif score >= 65:
        return "🟠 Quelques différences → mise à jour partielle recommandée"
    else:
        return "🔴 Catalogue obsolète → mise à jour nécessaire"


# =========================
# UI
# =========================
st.title("🔍 Comparateur fournisseur vs catalogue")
st.write("Analyse automatique des différences entre texte fournisseur et catalogue")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte fournisseur (actuel)", height=250)

with col2:
    text2 = st.text_area("📄 Texte catalogue (ancien)", height=250)

# =========================
# ACTION
# =========================
if st.button("Analyser"):

    if text1 and text2:

        with st.spinner("Analyse en cours..."):

            score, results = compare_texts(text1, text2)

            # sécurisation du score (IMPORTANT)
            score_value = float(score) if score is not None else 0
            score_value = max(0.0, min(score_value / 100, 1.0))

            # =========================
            # SCORE
            # =========================
            st.subheader("📊 Score de similarité global")

            st.metric("Similarité", f"{score:.2f}%")
            st.progress(score_value)

            # =========================
            # RECOMMANDATION
            # =========================
            st.subheader("🧠 Recommandation métier")
            st.info(recommendation(score))

            # =========================
            # DÉTAILS
            # =========================
            st.subheader("🔎 Analyse détaillée")

            st.dataframe(results, use_container_width=True)

    else:
        st.warning("Veuillez remplir les deux textes avant analyse")
