import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import matplotlib.pyplot as plt

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


def detect_changes(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return None, None, None, None

    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    matrix = np.zeros((len(s1), len(s2)))

    for i, e1 in enumerate(emb1):
        for j, e2 in enumerate(emb2):
            matrix[i][j] = cosine_similarity(e1, e2)

    # =========================
    # OPTIMAL MATCHING
    # =========================
    cost_matrix = 1 - matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_cols = set()

    results = []
    THRESHOLD = 0.55

    for i, j in zip(row_ind, col_ind):
        score = matrix[i][j]
        matched_cols.add(j)

        if score >= THRESHOLD:
            status = "🟢 OK"
        elif score >= 0.3:
            status = "🟠 Modifié"
        else:
            status = "🔴 Très différent"

        results.append({
            "fournisseur (Texte 1)": s1[i],
            "catalogue (Texte 2)": s2[j],
            "similarité (%)": round(score * 100, 2),
            "statut": status
        })

    # AJOUTS catalogue non matchés
    for j in range(len(s2)):
        if j not in matched_cols:
            results.append({
                "fournisseur (Texte 1)": "❌ absent",
                "catalogue (Texte 2)": s2[j],
                "similarité (%)": 0,
                "statut": "🟣 Ajout catalogue"
            })

    global_score = np.mean(matrix.max(axis=1)) * 100

    return global_score, results, matrix, s1, s2


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
st.write("Compare un texte fournisseur avec ton catalogue")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte 1 (Fournisseur)", height=250)

with col2:
    text2 = st.text_area("📄 Texte 2 (Catalogue)", height=250)


# =========================
# ACTION
# =========================
if st.button("Analyser les changements"):

    if text1 and text2:

        score, results, matrix, s1, s2 = detect_changes(text1, text2)

        # =========================
        # SCORE GLOBAL
        # =========================
        st.subheader("📊 Niveau d’alignement")
        st.metric("Alignement", f"{score:.2f}%")
        st.progress(min(score / 100, 1.0))

        # =========================
        # RECOMMANDATION
        # =========================
        st.subheader("🧠 Décision métier")
        st.info(recommendation(score))

        # =========================
        # HEATMAP
        # =========================
        st.subheader("🔥 Carte des similarités")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(matrix, annot=False, cmap="YlGnBu",
                    xticklabels=s2, yticklabels=s1, ax=ax)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)

        # =========================
        # TABLE RESULTATS
        # =========================
        st.subheader("🔎 Détail des différences")

        df = pd.DataFrame(results)

        st.dataframe(
            df.sort_values(by="similarité (%)", ascending=False),
            use_container_width=True
        )

        # =========================
        # ECARTS CRITIQUES
        # =========================
        st.subheader("🚨 Écarts critiques")

        critical = df[df["statut"].isin(["🔴 Très différent", "🟣 Ajout catalogue"])]

        if not critical.empty:
            st.dataframe(critical, use_container_width=True)
        else:
            st.success("Aucun écart critique détecté")

    else:
        st.warning("Veuillez remplir les deux textes.")
