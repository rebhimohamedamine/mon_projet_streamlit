"""
Application Streamlit pour la prédiction de variétés d'iris
"""
import streamlit as st
from prediction import predict
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Iris",
    page_icon="🌸",
    layout="wide"
)

# 1. Titre et description
st.title("🌸 Application de Prédiction d'Iris")
st.markdown("""
Cette application utilise un modèle de Machine Learning (RandomForest) pour prédire 
la variété d'une fleur d'iris en fonction de ses caractéristiques physiques.

**Variétés d'iris :**
- 🌺 **Setosa**
- 🌷 **Versicolor**
- 🌻 **Virginica**

---
""")

# 2. Sliders pour les caractéristiques (en deux colonnes)
st.subheader("📊 Entrez les caractéristiques de la fleur")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🍃 Caractéristiques des Sépales")
    sepal_length = st.slider(
        "Longueur du sépale (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1,
        help="Longueur du sépale en centimètres"
    )
    
    sepal_width = st.slider(
        "Largeur du sépale (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1,
        help="Largeur du sépale en centimètres"
    )

with col2:
    st.markdown("### 🌺 Caractéristiques des Pétales")
    petal_length = st.slider(
        "Longueur du pétale (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1,
        help="Longueur du pétale en centimètres"
    )
    
    petal_width = st.slider(
        "Largeur du pétale (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1,
        help="Largeur du pétale en centimètres"
    )

# Afficher les valeurs sélectionnées
st.markdown("---")
st.subheader("📝 Résumé des caractéristiques")

data_summary = pd.DataFrame({
    'Caractéristique': [
        'Longueur du sépale',
        'Largeur du sépale',
        'Longueur du pétale',
        'Largeur du pétale'
    ],
    'Valeur (cm)': [sepal_length, sepal_width, petal_length, petal_width]
})

st.table(data_summary)

# 3. Bouton de prédiction
st.markdown("---")

if st.button("🔮 Prédire la variété", type="primary", use_container_width=True):
    try:
        # Faire la prédiction
        with st.spinner("Analyse en cours..."):
            predicted_class, probabilities = predict(
                sepal_length, sepal_width, petal_length, petal_width
            )
        
        # Afficher les résultats
        st.success("✅ Prédiction réussie!")
        
        st.markdown("---")
        st.subheader("🎯 Résultats de la prédiction")
        
        # Afficher la classe prédite avec une grande mise en forme
        st.markdown(f"""
        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">Variété prédite : {predicted_class.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Afficher les probabilités
        st.subheader("📊 Probabilités pour chaque variété")
        
        prob_df = pd.DataFrame({
            'Variété': list(probabilities.keys()),
            'Probabilité': [f"{prob*100:.2f}%" for prob in probabilities.values()],
            'Confiance': list(probabilities.values())
        })
        
        # Trier par probabilité décroissante
        prob_df = prob_df.sort_values('Confiance', ascending=False)
        
        # Afficher sous forme de barres
        st.bar_chart(prob_df.set_index('Variété')['Confiance'])
        
        # Afficher le tableau
        st.table(prob_df[['Variété', 'Probabilité']])
        
        # Message de confiance
        max_prob = max(probabilities.values())
        if max_prob > 0.9:
            st.info("🎯 Le modèle est très confiant dans cette prédiction!")
        elif max_prob > 0.7:
            st.info("✅ Le modèle est assez confiant dans cette prédiction.")
        else:
            st.warning("⚠️ Le modèle est moins certain de cette prédiction.")
            
    except FileNotFoundError as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.info("💡 Astuce : Exécutez d'abord `python model.py` pour entraîner le modèle.")
    except Exception as e:
        st.error(f"❌ Une erreur s'est produite : {str(e)}")

# Sidebar avec informations
st.sidebar.title("ℹ️ Informations")
st.sidebar.markdown("""
### À propos du dataset Iris

Le dataset Iris est un ensemble de données célèbre 
en Machine Learning contenant 150 échantillons de fleurs d'iris.

**Caractéristiques :**
- 4 attributs (longueur et largeur des sépales et pétales)
- 3 classes (Setosa, Versicolor, Virginica)
- 50 échantillons par classe

### Modèle utilisé

**RandomForest Classifier**
- Algorithme d'ensemble
- Haute précision
- Robuste au surapprentissage
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**TP1 MLOps - Streamlit**")
st.sidebar.markdown("*Dr. ASMA MEKKI*")
st.sidebar.markdown("*IDS 5*")