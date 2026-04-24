import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# =========================
# CONFIG GPT
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

model = SentenceTransformer("all-MiniLM-L6-v2")


def split_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def build_diff_data(text1, text2):

    s1 = split_sentences(text1)
    s2 = split_sentences(text2)

    e1 = model.encode(s1)
    e2 = model.encode(s2)

    matches = []

    for i, emb in enumerate(e1):
        sims = [cosine(emb, e) for e in e2]
        best = int(np.argmax(sims))
        score = float(max(sims))

        matches.append({
            "texte_1": s1[i],
            "texte_2": s2[best] if score > 0.3 else "ABSENT",
            "similarite": round(score * 100, 2)
        })

    return matches


# =========================
# GPT ANALYSIS
# =========================
def gpt_analyze(matches):

    prompt = f"""
Tu es un expert en contenu touristique.

Analyse les différences entre ces deux textes :

Texte 1 = fournisseur (source actuelle)
Texte 2 = catalogue client (version ancienne)

Voici les correspondances :
{matches}

Réponds en français avec :

1. 🟢 Résumé global
2. 🔵 Différences principales (structurées)
3. ⚠️ Éléments manquants ou modifiés
4. 🧭 Conclusion métier (le catalogue doit-il être mis à jour ?)

Sois clair, structuré, et orienté métier tourisme.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un expert en analyse de contenus touristiques."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
