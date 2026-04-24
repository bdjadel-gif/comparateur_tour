import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Comparateur fournisseur / catalogue",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# NLP UTILS
# =========================
def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def explain_difference(score):
    if score > 0.85:
        return "Très proche sémantiquement"
    elif score > 0.65:
        return "Légère reformulation ou détail modifié"
    elif score > 0.4:
        return "Différence notable de formulation ou contenu"
    else:
        return "Contenu divergent ou ajout/suppression important"


# =========================
# CORE ANALYSIS
# =========================
def analyze(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return None

    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    # Similarity matrix
    matrix = np.zeros((len(s1), len(s2)))

    for i in range(len(s1)):
        for j in range(len(s2)):
            matrix[i][j] = cosine_sim(emb1[i], emb2[j])

    # =========================
    # GLOBAL SCORE
    # =========================
    global_score = np.mean(matrix.max(axis=1)) * 100

    # =========================
    # OPTIMAL MATCHING
    # =========================
    cost = 1 - matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    used_cols = set()
    results = []

    THRESHOLD = 0.55

    # Matches optimaux
    for i, j in zip(row_ind, col_ind):
        score = matrix[i][j]
        used_cols.add(j)

        status = (
            "🟢 OK" if score > 0.75 else
            "🟠 Modifié" if score > THRESHOLD else
            "🔴 Divergent"
        )

        results.append({
            "phrase catalogue": s2[j],
            "phrase fournisseur": s1[i],
            "similarité (%)": round(score * 100, 2),
            "statut": status,
            "explication": explain_difference(score)
        })

    # =========================
    # AJOUTS CATALOGUE
    # =========================
    for j in range(len(s2)):
        if j not in used_cols:
            results.append({
                "phrase catalogue": s2[j],
                "phrase fournisseur": "❌ absent",
                "similarité (%)": 0,
                "statut": "🟣 Ajout catalogue",
                "explication": "Nouvelle information absente du texte fournisseur"
            })

    return global_score, results, matrix, s1, s2


# =========================
# UI
# =========================
st.title("🔍 Comparateur fournisseur / catalogue sémantique")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte fournisseur (référence)", height=250)

with col2:
    text2 = st.text_area("📄 Texte catalogue (interne)", height=250)


# =========================
# RUN
# =========================
if st.button("Analyser"):

    if text1 and text2:

        score, results, matrix, s1, s2 = analyze(text1, text2)

        # =========================
        # SCORE GLOBAL
        # =========================
        st.subheader("📊 Score de compatibilité")
        st.metric("Compatibilité", f"{score:.2f}%")
        st.progress(min(score / 100, 1.0))

        # =========================
        # HEATMAP
        # =========================
        st.subheader("🔥 Matrice de similarité")

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, cmap="viridis")

        ax.set_xticks(np.arange(len(s2)))
        ax.set_yticks(np.arange(len(s1)))

        ax.set_xticklabels(s2, rotation=45, ha="right")
        ax.set_yticklabels(s1)

        plt.colorbar(im, ax=ax)

        st.pyplot(fig)

        # =========================
        # TABLE RESULTS
        # =========================
        st.subheader("📋 Analyse détaillée")

        df = pd.DataFrame(results)

        st.dataframe(
            df.sort_values(by="similarité (%)", ascending=False),
            use_container_width=True
        )

        # =========================
        # PROBLEMES CRITIQUES
        # =========================
        st.subheader("🚨 Écarts critiques")

        critical = df[df["statut"].isin(["🔴 Divergent", "🟣 Ajout catalogue"])]

        if not critical.empty:
            st.dataframe(critical, use_container_width=True)
        else:
            st.success("Aucun écart critique détecté")

    else:
        st.warning("Veuillez remplir les deux textes")
