import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import openai

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Analyse changements fournisseur",
    page_icon="🔍",
    layout="wide"
)

# =========================
# OPENAI KEY
# =========================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# =========================
# NLP MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# UTILS NLP
# =========================
def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def build_matches(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    if not s1 or not s2:
        return []

    e1 = model.encode(s1)
    e2 = model.encode(s2)

    matches = []

    for i, emb in enumerate(e1):
        sims = [cosine(emb, e) for e in e2]

        best_idx = int(np.argmax(sims))
        best_score = float(max(sims))

        matches.append({
            "texte_1": s1[i],
            "texte_2": s2[best_idx] if best_score > 0.3 else "❌ absent",
            "similarité (%)": round(best_score * 100, 2)
        })

    return matches


# =========================
# GPT ANALYSIS (VERSION COMPATIBLE)
# =========================
def gpt_analyze(matches):

    prompt = f"""
Tu es un expert en analyse de contenus touristiques.

Compare ces deux versions :
- Texte 1 = fournisseur (actuel)
- Texte 2 = catalogue client (ancien)

Voici les correspondances :
{matches}

Réponds en français avec :

🟢 1. Résumé global
🔵 2. Différences principales
⚠️ 3. Éléments manquants ou modifiés
🧭 4. Conclusion métier (mise à jour nécessaire ou non)

Sois clair et structuré.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un expert en analyse de contenus touristiques."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response["choices"][0]["message"]["content"]


# =========================
# UI
# =========================
st.title("🔍 Analyse intelligente des changements fournisseur")
st.write("Compare un texte fournisseur (Texte 1) avec ton catalogue (Texte 2)")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📄 Texte 1 (Fournisseur)", height=250)

with col2:
    text2 = st.text_area("📄 Texte 2 (Catalogue)", height=250)


# =========================
# ACTION
# =========================
if st.button("Analyser les différences"):

    if text1 and text2:

        with st.spinner("Analyse en cours..."):

            matches = build_matches(text1, text2)

            if not matches:
                st.warning("Textes insuffisants pour analyse.")
                st.stop()

            # SCORE GLOBAL
            global_score = np.mean([m["similarité (%)"] for m in matches])

            st.subheader("📊 Niveau d’alignement")
            st.metric("Similarité globale", f"{global_score:.2f}%")
            st.progress(min(global_score / 100, 1.0))

            # GPT ANALYSIS
            st.subheader("🧠 Analyse IA (GPT)")
            analysis = gpt_analyze(matches)
            st.markdown(analysis)

            # TABLE
            st.subheader("🔎 Détails comparatifs")
            st.dataframe(matches, use_container_width=True)

    else:
        st.warning("Veuillez remplir les deux textes.")
