import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Comparateur de descriptifs IA", layout="centered")

st.title("🔍 Analyse de similarité entre deux descriptifs")

st.write("Collez deux descriptifs pour analyser leur similarité et leurs différences.")

# Inputs
desc1 = st.text_area("📄 Descriptif 1", height=200)
desc2 = st.text_area("📄 Descriptif 2", height=200)

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="french")
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

def extract_differences(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    only_in_1 = sorted(list(set1 - set2))
    only_in_2 = sorted(list(set2 - set1))

    return only_in_1, only_in_2

if st.button("🚀 Lancer l'analyse"):

    if not desc1 or not desc2:
        st.warning("Merci de remplir les deux descriptifs.")
    else:
        # Similarité
        score = compute_similarity(desc1, desc2)

        st.subheader("📊 Score de similarité")
        st.metric(label="Similarité (cosine TF-IDF)", value=f"{score*100:.2f} %")

        # Interprétation simple
        if score > 0.8:
            st.success("Très forte similarité")
        elif score > 0.5:
            st.info("Similarité modérée")
        else:
            st.warning("Faible similarité")

        # Différences
        st.subheader("🧩 Différences détectées")

        diff1, diff2 = extract_differences(desc1, desc2)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 Uniquement dans le descriptif 1")
            st.write(diff1[:200])

        with col2:
            st.markdown("### 🟢 Uniquement dans le descriptif 2")
            st.write(diff2[:200])

        st.caption("⚠️ Analyse basique par tokens. Pour une IA plus avancée, on peut intégrer spaCy ou embeddings OpenAI.")
