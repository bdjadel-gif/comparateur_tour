import streamlit as st

st.set_page_config(page_title="Comparateur d'activités", layout="wide")

st.title("🔍 Comparateur de descriptifs d'activités")

text1 = st.text_area("Descriptif 1", height=300)
text2 = st.text_area("Descriptif 2", height=300)

def extract_info(text):
    text = text.lower()
    
    data = {
        "transport": "oui" if "hôtel" in text or "prise en charge" in text else "non",
        "repas": "oui" if "déjeuner" in text else "non",
        "durée": "mentionnée" if "heure" in text or "durée" in text else "non précisée",
        "lieux": []
    }
    
    lieux_possibles = ["kingston", "trenchtown", "orange street", "museum", "parc"]
    
    for lieu in lieux_possibles:
        if lieu in text:
            data["lieux"].append(lieu)
    
    return data

if st.button("Comparer"):
    if text1 and text2:
        data1 = extract_info(text1)
        data2 = extract_info(text2)
        
        st.subheader("📊 Comparaison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Descriptif 1")
            st.json(data1)
        
        with col2:
            st.write("### Descriptif 2")
            st.json(data2)
        
        st.subheader("⚠️ Différences")
        
        differences = []
        
        for key in data1:
            if data1[key] != data2[key]:
                differences.append(f"{key} : {data1[key]} vs {data2[key]}")
        
        if differences:
            for diff in differences:
                st.write("- " + diff)
        else:
            st.success("Aucune différence détectée")
    
    else:
        st.warning("Merci de remplir les deux descriptifs")
