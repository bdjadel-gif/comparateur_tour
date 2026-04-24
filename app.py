import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Comparateur de textes",
    page_icon="🔍",
    layout="wide"
)

# =========================
# MODEL (IA simple)
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


def compare(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return 0, []

    e1 = model.encode(s1)
    e2 = model.encode(s2)

    results = []

    for i, emb in enumerate(e1):
        sims = [cosine(emb, e) for e in e2]
        best = max(sims)

        results.append({
            "texte 1": s1[i],
            "similarité (%)": round(best * 100, 2)
        })

    global_score = np.mean([r["similarité (%)"] for r in results])

    return global_score, results


# =========================
# UI
# =========================
st.title("🔍 Comparateur de textes touristiques simple")

text1 = st.text_area("📄 Texte 1 (Fournisseur)", height=200)
text2 = st.text_area("📄 Texte 2 (Catalogue)", height=200)

if st.button("Comparer"):

    if text1 and text2:

        score, results = compare(text1, text2)

        st.subheader("📊 Score global")
        st.metric("Similarité", f"{score:.2f}%")

        score_value = float(score) if score is not None else 0
score_value = max(0.0, min(score_value / 100, 1.0))

st.progress(score_value)

        st.subheader("🔎 Détails")

        st.dataframe(results)

    else:
        st.warning("Remplis les deux textes")
