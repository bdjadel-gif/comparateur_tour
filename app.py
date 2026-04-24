import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# ---------------- INIT ----------------
st.set_page_config(page_title="Comparateur IA intelligent", layout="wide")

st.title("🧠 Comparateur IA intelligent (niveau métier tourisme)")
st.write("Compare texte fournisseur (vérité) vs catalogue marketing")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- PARAMÈTRES MÉTIER ----------------
KEYWORDS_IMPORTANTS = [
    "musée", "museum", "palais", "palace", "église", "church",
    "tour", "tower", "place", "piazza", "monument", "arènes",
    "balcon", "basilique", "cathédrale", "front de mer"
]

WEIGHTS = {
    "important": 1.5,
    "normal": 1.0,
    "marketing": 0.7
}

# ---------------- UTILS ----------------
def split_blocks(text):
    return [b.strip() for b in re.split(r'\n+', text) if len(b.strip()) > 20]

def embed(texts):
    return model.encode(texts)

def is_important(text):
    return any(k in text.lower() for k in KEYWORDS_IMPORTANTS)

def block_weight(text):
    if is_important(text):
        return WEIGHTS["important"]
    return WEIGHTS["normal"]

# ---------------- SIMILARITÉ ----------------
def compute_similarity(f_blocks, c_blocks):
    emb_f = embed(f_blocks)
    emb_c = embed(c_blocks)

    sim = cosine_similarity(emb_f, emb_c)

    best_scores = sim.max(axis=1)

    weighted_scores = []
    for i, score in enumerate(best_scores):
        w = block_weight(f_blocks[i])
        weighted_scores.append(score * w)

    return best_scores, weighted_scores

# ---------------- AJOUTS / OMISSIONS ----------------
def detect_gaps(f_blocks, c_blocks):
    emb_f = embed(f_blocks)
    emb_c = embed(c_blocks)

    sim = cosine_similarity(emb_c, emb_f)

    omissions = []
    for i, row in enumerate(sim.T):
        if max(row) < 0.55:
            omissions.append(f_blocks[i])

    sim2 = cosine_similarity(emb_f, emb_c)

    ajouts = []
    for i, row in enumerate(sim2.T):
        if max(row) < 0.55:
            ajouts.append(c_blocks[i])

    return omissions, ajouts

# ---------------- ANALYSE PRINCIPALE ----------------
def analyze(fournisseur, catalogue):

    f_blocks = split_blocks(fournisseur)
    c_blocks = split_blocks(catalogue)

    base_scores, weighted_scores = compute_similarity(f_blocks, c_blocks)

    omissions, ajouts = detect_gaps(f_blocks, c_blocks)

    # SCORE FINAL HYBRIDE
    semantic_score = np.mean(weighted_scores) * 100
    coverage_penalty = (len(omissions) / max(len(f_blocks), 1)) * 20
    bonus_penalty = (len(ajouts) / max(len(c_blocks), 1)) * 10

    final_score = semantic_score - coverage_penalty - bonus_penalty
    final_score = max(0, min(100, final_score))

    # LABEL
    if final_score >= 95:
        label = "✅ Réécriture fidèle (niveau parfait)"
    elif final_score >= 85:
        label = "🟢 Très proche"
    elif final_score >= 70:
        label = "🟡 Différences modérées"
    elif final_score >= 50:
        label = "🟠 Différences importantes"
    else:
        label = "🔴 Programme différent"

    # correspondances fortes
    correspondances = [
        f_blocks[i] for i, s in enumerate(base_scores) if s > 0.75
    ]

    modifications = [
        f_blocks[i] for i, s in enumerate(base_scores) if 0.55 <= s <= 0.75
    ]

    return final_score, label, f_blocks, correspondances, modifications, omissions, ajouts

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    fournisseur = st.text_area("📄 Texte fournisseur", height=300)

with col2:
    catalogue = st.text_area("📝 Texte catalogue", height=300)

if st.button("🔍 Analyser intelligemment"):

    if fournisseur and catalogue:

        score, label, structure, corr, modif, omis, ajouts = analyze(fournisseur, catalogue)

        st.subheader(f"📊 Score intelligent : {round(score,2)}%")
        st.markdown(f"### {label}")

        st.progress(score / 100)

        st.divider()

        st.subheader("🧩 Structure fournisseur")
        for s in structure:
            st.write("•", s)

        st.subheader("✅ Correspondances fortes")
        for c in corr:
            st.success(c)

        st.subheader("⚠️ Modifications")
        for m in modif:
            st.warning(m)

        st.subheader("❌ Omissions (critique fournisseur)")
        for o in omis:
            st.error(o)

        st.subheader("➕ Ajouts catalogue")
        for a in ajouts:
            st.info(a)

    else:
        st.warning("Veuillez remplir les deux textes")
