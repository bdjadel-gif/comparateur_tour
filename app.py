import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# ---------------- INIT ----------------
st.set_page_config(page_title="Comparateur IA tourisme", layout="wide")

st.title("🧠 Comparateur IA de textes touristiques")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- KEYWORDS MÉTIER ----------------
KEYWORDS = [
    "musée", "palais", "tour", "église", "basilique",
    "place", "monument", "pont", "château",
    "centre-ville", "balcon", "arène", "galerie"
]

# ---------------- UTILS ----------------
def split_into_blocks(text):
    blocks = re.split(r'\n+', text)
    return [b.strip() for b in blocks if len(b.strip()) > 25]

def embed(text_list):
    return model.encode(text_list)

def keyword_boost(text):
    boost = 1.0
    for k in KEYWORDS:
        if k.lower() in text.lower():
            boost += 0.05
    return boost

def compute_similarity(blocks_f, blocks_c):
    emb_f = embed(blocks_f)
    emb_c = embed(blocks_c)
    sim_matrix = cosine_similarity(emb_f, emb_c)
    best_scores = sim_matrix.max(axis=1)
    return best_scores

# ---------------- ANALYSE ----------------
def analyze(fournisseur, catalogue):

    blocks_f = split_into_blocks(fournisseur)
    blocks_c = split_into_blocks(catalogue)

    scores = compute_similarity(blocks_f, blocks_c)

    weighted_scores = []
    correspondances = []
    modifications = []
    omissions = []

    # -------- comparaison fournisseur -> catalogue --------
    for i, score in enumerate(scores):
        text = blocks_f[i]
        boost = keyword_boost(text)
        final_score = score * boost

        weighted_scores.append(final_score)

        if final_score > 0.75:
            correspondances.append(text)
        elif final_score > 0.5:
            modifications.append(text)
        else:
            omissions.append(text)

    # -------- ajouts catalogue --------
    scores_c = compute_similarity(blocks_c, blocks_f)

    ajouts = []
    for i, score in enumerate(scores_c):
        if score < 0.55:
            ajouts.append(blocks_c[i])

    # -------- SCORE FINAL --------
    base_score = np.mean(weighted_scores) * 100
    adjusted_score = (base_score * 0.85) + 15
    adjusted_score = min(98, max(40, adjusted_score))

    # -------- LABEL --------
    if adjusted_score >= 92:
        label = "🟢 Réécriture fidèle"
    elif adjusted_score >= 80:
        label = "🟡 Légères différences"
    elif adjusted_score >= 65:
        label = "🟠 Différences notables"
    else:
        label = "🔴 Programme différent"

    # -------- JUSTIFICATION --------
    nb_omissions = len(omissions)
    nb_ajouts = len(ajouts)
    nb_modifs = len(modifications)

    if adjusted_score >= 90:
        explanation = (
            "Le score est très élevé car le programme est strictement identique entre les deux versions. "
            "Les différences sont uniquement stylistiques et n'affectent pas les activités ou les lieux."
        )

    elif adjusted_score >= 80:
        explanation = (
            "Le score est élevé avec de légères différences. "
            f"{nb_modifs} reformulation(s) mineure(s) et {nb_ajouts} ajout(s) léger(s) ont été détectés, "
            "sans impact majeur sur le programme."
        )

    elif adjusted_score >= 65:
        explanation = (
            "Le score est moyen. "
            f"{nb_omissions} élément(s) du programme fournisseur sont absents et "
            f"{nb_ajouts} ajout(s) apparaissent dans le catalogue. "
            "Certaines étapes ont été modifiées ou reformulées."
        )

    else:
        explanation = (
            "Le score est faible. "
            f"{nb_omissions} éléments du programme fournisseur sont manquants et "
            f"{nb_ajouts} éléments ont été ajoutés dans le catalogue, "
            "indiquant une divergence importante du contenu."
        )

    return adjusted_score, label, blocks_f, correspondances, modifications, omissions, ajouts, explanation

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    fournisseur = st.text_area("📄 Texte fournisseur (vérité)", height=300)

with col2:
    catalogue = st.text_area("📝 Texte catalogue", height=300)

if st.button("🔍 Comparer les textes"):

    if fournisseur and catalogue:

        score, label, structure, corr, modif, omis, ajouts, explanation = analyze(fournisseur, catalogue)

        st.subheader(f"📊 Score de compatibilité : {round(score,2)}%")
        st.markdown(f"### {label}")

        st.progress(int(score))

        # -------- JUSTIFICATION --------
        st.subheader("🔍 Justification du score")
        st.write(explanation)

        st.divider()

        # -------- STRUCTURE --------
        st.subheader("🧩 Structure fournisseur")
        for s in structure:
            st.write("•", s)

        st.subheader("✅ Correspondances")
        for c in corr:
            st.write("•", c)

        st.subheader("⚠️ Modifications")
        for m in modif:
            st.write("•", m)

        st.subheader("❌ Omissions")
        for o in omis:
            st.write("•", o)

        st.subheader("➕ Ajouts")
        for a in ajouts:
            st.write("•", a)

    else:
        st.warning("Veuillez remplir les deux textes")
