import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Détection de changements fournisseur",
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


def compute_matrix(sentences1, sentences2):
    emb1 = model.encode(sentences1)
    emb2 = model.encode(sentences2)

    matrix = np.zeros((len(sentences1), len(sentences2)))

    for i, e1 in enumerate(emb1):
        for j, e2 in enumerate(emb2):
            matrix[i][j] = cosine_similarity(e1, e2)

    return matrix


def detect_changes(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return None, None, None

    matrix = compute_matrix(s1, s2)

    results = []

    for i in range(len(s1)):
        best_j = np.argmax(matrix[i])
        best_score = matrix[i][best_j]

        results.append({
            "fournisseur (Texte 1)": s1[i],
            "catalogue (Texte 2)": s2[best_j] if best_score > 0.3 else "❌ absent",
            "similarité (%)": round(best_score * 100, 2)
        })

    global_score = np.mean(matrix.max(axis=1)) * 100

    return global_score, results, matrix


def recommendation(score):

    if score > 85:
        return "🟢 Catalogue à jour (aligné avec le fournisseur)"
    elif score > 65:
        return "🟠 Écarts détectés → mise à jour partielle recommandée"
    else:
        return "🔴 Catalogue obsolète → mise à jour urgente requise"


# =========================
# UI
# =========================
st.title("🔍 Détection de changements fournisseur")
st.write("Compare un texte fournisseur (Texte 1) avec ton catalogue (Texte 2)")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte 1 (Fournisseur - source actuelle)", height=250)

with col2:
    text2 = st.text_area("📄 Texte 2 (Ton catalogue)", height=250)


# =========================
# ACTION
# =========================
if st.button("Analyser les changements"):

    if text1 and text2:

        score, results, matrix = detect_changes(text1, text2)

        # =========================
        # 1. SCORE GLOBAL
        # =========================
        st.subheader("📊 Niveau d’alignement avec le fournisseur")
        st.metric("Alignement", f"{score:.2f}%")

        st.progress(min(score / 100, 1.0))

        # =========================
        # 2. RECOMMANDATION
        # =========================
        st.subheader("🧠 Décision métier")
        st.info(recommendation(score))

        # =========================
        # 3. TABLE COMPARATIVE
        # =========================
        st.subheader("🔎 Changements détectés")

        df = pd.DataFrame(results)

        def highlight(score):
            if score > 75:
                return "🟢 OK"
            elif score > 50:
                return "🟠 Différence"
            else:
                return "🔴 Problème"

        df["statut"] = df["similarité (%)"].apply(highlight)

        st.dataframe(df, use_container_width=True)

        # =========================
        # 4. ALERTES
        # =========================
        st.subheader("⚠️ Points d’attention")

        obsolete = df[df["catalogue (Texte 2)"] == "❌ absent"]

        if len(obsolete) > 0:
            st.warning("Certaines informations du fournisseur ne sont pas présentes dans ton catalogue.")
            st.dataframe(obsolete)

    else:
        st.warning("Veuillez remplir les deux textes.")
