import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# --- Model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utils ---
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

def compare_texts(text1, text2):
    sent1 = split_sentences(text1)
    sent2 = split_sentences(text2)

    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)

    results = []

    for i, e1 in enumerate(emb1):
        sims = [cosine_similarity(e1, e2) for e2 in emb2]
        best = max(sims) if sims else 0

        if best > 0.75:
            label = "🟢 similaire"
        elif best > 0.4:
            label = "🟠 partiellement similaire"
        else:
            label = "🔴 différent"

        results.append({
            "sentence": sent1[i],
            "score": round(best * 100, 2),
            "label": label
        })

    # global score
    global_score = np.mean([r["score"] for r in results]) if results else 0

    return global_score, results

# --- UI ---
st.title("🔍 Comparateur de textes touristiques")

st.write("Compare deux descriptions d'activités touristiques et analyse leurs similarités.")

text1 = st.text_area("Texte 1")
text2 = st.text_area("Texte 2")

if st.button("Comparer"):

    if text1 and text2:

        score, results = compare_texts(text1, text2)

        st.subheader(f"📊 Score global de similarité : {round(score, 2)}%")

        st.divider()

        st.subheader("🔎 Analyse détaillée")

        for r in results:
            st.write(f"{r['label']} — {r['sentence']} ({r['score']}%)")

    else:
        st.warning("Veuillez entrer deux textes.")
