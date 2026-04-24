import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --------- INIT ---------
st.set_page_config(page_title="Comparateur de textes touristiques", layout="wide")

st.title("🧠 Comparateur de textes touristiques")
st.write("Analyse de compatibilité entre texte fournisseur (source de vérité) et texte catalogue")

model = SentenceTransformer('all-MiniLM-L6-v2')

# --------- UTILS ---------
def split_into_blocks(text):
    blocks = re.split(r'\n+|\.\s+', text)
    return [b.strip() for b in blocks if len(b.strip()) > 20]

def embed(text_list):
    return model.encode(text_list)

def compute_similarity(blocks1, blocks2):
    emb1 = embed(blocks1)
    emb2 = embed(blocks2)
    sim_matrix = cosine_similarity(emb1, emb2)
    best_scores = sim_matrix.max(axis=1)
    return best_scores, sim_matrix

# --------- ANALYSE ---------
def analyze(fournisseur, catalogue):
    blocks_f = split_into_blocks(fournisseur)
    blocks_c = split_into_blocks(catalogue)

    scores, sim_matrix = compute_similarity(blocks_f, blocks_c)
    avg_score = float(np.mean(scores)) * 100

    correspondances = []
    omissions = []
    modifications = []

    for i, score in enumerate(scores):
        if score > 0.75:
            correspondances.append(blocks_f[i])
        elif score > 0.5:
            modifications.append(blocks_f[i])
        else:
            omissions.append(blocks_f[i])

    # Détection des ajouts
    scores_c, _ = compute_similarity(blocks_c, blocks_f)
    ajouts = []
    for i, score in enumerate(scores_c):
        if score < 0.5:
            ajouts.append(blocks_c[i])

    # Label métier
    if avg_score >= 95:
        label = "✅ Réécriture fidèle"
    elif avg_score >= 80:
        label = "🟡 Légères différences"
    elif avg_score >= 60:
        label = "🟠 Différences notables"
    else:
        label = "🔴 Programme différent"

    return avg_score, label, blocks_f, correspondances, modifications, omissions, ajouts

# --------- UI ---------
col1, col2 = st.columns(2)

with col1:
    fournisseur = st.text_area("📄 Texte fournisseur (source de vérité)", height=300)

with col2:
    catalogue = st.text_area("📝 Texte catalogue", height=300)

if st.button("🔍 Comparer les textes"):

    if fournisseur and catalogue:
        score, label, structure, corr, modif, omis, ajouts = analyze(fournisseur, catalogue)

        st.subheader(f"📊 Score de compatibilité : {round(score,2)}%")
        st.markdown(f"### {label}")

        st.divider()

        st.subheader("🧩 Structure commune")
        for s in structure:
            st.write(f"- {s}")

        st.subheader("✅ Correspondances fortes")
        for c in corr:
            st.write(f"- {c}")

        st.subheader("⚠️ Modifications")
        for m in modif:
            st.write(f"- {m}")

        st.subheader("❌ Omissions")
        for o in omis:
            st.write(f"- {o}")

        st.subheader("➕ Ajouts")
        for a in ajouts:
            st.write(f"- {a}")

    else:
        st.warning("Veuillez remplir les deux textes")
