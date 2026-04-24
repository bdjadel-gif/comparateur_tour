import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
from scipy.optimize import linear_sum_assignment

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
# NLP FUNCTIONS
# =========================
def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def explain(score):
    if score > 0.85:
        return "Très proche"
    elif score > 0.65:
        return "Légère différence"
    elif score > 0.4:
        return "Différence notable"
    else:
        return "Divergence forte"


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

    matrix = np.zeros((len(s1), len(s2)))

    for i in range(len(s1)):
        for j in range(len(s2)):
            matrix[i][j] = cosine_sim(emb1[i], emb2[j])

    # =========================
    # SCORE GLOBAL
    # =========================
    global_score = np.mean(matrix.max(axis=1)) * 100

    # =========================
    # MATCHING OPTIMAL
    # =========================
    cost = 1 - matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    used_cols = set()
    results = []

    THRESHOLD = 0.55

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
            "similarité": round(score * 100, 2),
            "statut": status
        })

    # Ajouts catalogue
    for j in range(len(s2)):
        if j not in used_cols:
            results.append({
                "phrase catalogue": s2[j],
                "phrase fournisseur": "❌ absent",
                "similarité": 0,
                "statut": "🟣 Ajout catalogue"
            })

    return global_score, results, s1, s2


# =========================
# TEXT GENERATION (NOUVEAU)
# =========================
def build_improved_catalog(results):

    improved = []

    for r in results:

        cat = r["phrase catalogue"]
        sup = r["phrase fournisseur"]
        score = r["similarité"] / 100

        # Ajout catalogue → on ignore dans version alignée
        if sup == "❌ absent":
            continue

        # Très bon match → on garde fournisseur
        if score >= 0.75:
            improved.append(sup)

        # Modifié → on privilégie fournisseur (version corrigée)
        elif score >= 0.55:
            improved.append(sup)

        # Divergent → on signale correction
        else:
            improved.append(f"[À vérifier] {sup}")

    return ". ".join(improved) + "."


# =========================
# UI
# =========================
st.title("🔍 Comparateur fournisseur / catalogue intelligent")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte fournisseur", height=250)

with col2:
    text2 = st.text_area("📄 Texte catalogue", height=250)


# =========================
# RUN
# =========================
if st.button("Analyser"):

    if text1 and text2:

        score, results, s1, s2 = analyze(text1, text2)

        # =========================
        # SCORE
        # =========================
        st.subheader("📊 Score de compatibilité")
        st.metric("Compatibilité", f"{score:.2f}%")
        st.progress(min(score / 100, 1.0))

        # =========================
        # TABLE
        # =========================
        st.subheader("📋 Analyse détaillée")

        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True)

        # =========================
        # 🔥 NOUVEAU : TEXTE AMÉLIORÉ
        # =========================
        st.subheader("✍️ Proposition de catalogue amélioré")

        improved_text = build_improved_catalog(results)

        st.info("Version du catalogue alignée avec le fournisseur :")
        st.write(improved_text)

    else:
        st.warning("Veuillez remplir les deux textes")
