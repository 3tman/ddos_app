import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Détection DDoS AI", layout="wide")

# ==========================================
# SYSTÈME DE CONNEXION
# ==========================================
def login():
    st.title("🔐 Connexion au Système de Sécurité")
    with st.form("login_form"):
        username = st.text_input("Utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter"):
            if username == "admin" and password == "admin123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Identifiants invalides")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
    st.stop()

# ==========================================
# FONCTION DE CHARGEMENT
# ==========================================

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    
    # NOUVEAU : On transforme tout le texte en nombres automatiquement !
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    target_col = 'label' if 'label' in df.columns else 'target'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y, df, target_col

# ==========================================
# BARRE LATÉRALE ET NAVIGATION
# ==========================================
st.sidebar.title("🛡️ Dashboard DDoS")
if st.sidebar.button("Déconnexion"):
    st.session_state.clear()
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Charger le dataset CSV", type=["csv"])

menu = st.sidebar.radio(
    "Menu principal",
    ["Exploration (EDA)", "Prétraitement & SMOTE", "Entraînement Modèle", "🔮 Test de Prédiction"]
)

# ==========================================
# LOGIQUE DES ONGLETS
# ==========================================
if uploaded_file:
    X, y, df_full, target_col = load_and_preprocess(uploaded_file)

    # --- ONGLET 1 : EDA ---
    if menu == "Exploration (EDA)":
        st.title("📊 Analyse des données")
        st.write("Aperçu des premières lignes :")
        st.dataframe(df_full.head())
        
        st.write("Distribution des classes (0 = Normal, 1 = DDoS) :")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x=y, ax=ax, palette="viridis")
        st.pyplot(fig)

    # --- ONGLET 2 : SMOTE ---
    elif menu == "Prétraitement & SMOTE":
        st.title("⚙️ Équilibrage des classes avec SMOTE")
        st.write(f"Distribution avant SMOTE :\n{y.value_counts()}")
        
        if st.button("Lancer SMOTE"):
            with st.spinner("Équilibrage en cours..."):
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                st.session_state['X_res'] = X_res
                st.session_state['y_res'] = y_res
                st.success("Données équilibrées avec succès !")
                st.write(f"Nouvelle distribution :\n{y_res.value_counts()}")

    # --- ONGLET 3 : ENTRAÎNEMENT ---
    elif menu == "Entraînement Modèle":
        st.title("🤖 Entraînement du Modèle (Random Forest)")
        if 'X_res' not in st.session_state:
            st.warning("⚠️ Passez d'abord par l'onglet 'Prétraitement & SMOTE' pour équilibrer les données.")
        else:
            if st.button("Démarrer l'apprentissage"):
                with st.spinner("Entraînement du modèle en cours..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state['X_res'], st.session_state['y_res'], test_size=0.2, random_state=42
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Sauvegarde dans la session pour la prédiction
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = X.columns.tolist()
                    st.session_state['df_full'] = df_full # Sauvegarde pour le test aléatoire
                    
                    acc = accuracy_score(y_test, model.predict(X_test_scaled))
                    st.success(f"✅ Modèle entraîné et sauvegardé ! Précision sur les données de test : {acc:.2%}")

    # --- ONGLET 4 : TEST DE PRÉDICTION ---
    elif menu == "🔮 Test de Prédiction":
        st.title("🚀 Tester le Modèle")
        
        if 'model' not in st.session_state:
            st.error("❌ Vous devez d'abord entraîner le modèle dans l'onglet 'Entraînement Modèle'.")
        else:
            type_test = st.radio("Choisissez le mode de test :", ["Test Rapide (Ligne aléatoire du dataset)", "Saisie Manuelle (Entrer vos propres valeurs)"])
            
            # MODE 1 : ALÉATOIRE
            if type_test == "Test Rapide (Ligne aléatoire du dataset)":
                st.write("Ce mode sélectionne une ligne au hasard dans votre dataset original et demande à l'IA de la prédire.")
                if st.button("Tirer une ligne au hasard et Prédire"):
                    # Tirer une ligne au hasard
                    random_row = st.session_state['df_full'].sample(1)
                    true_label = random_row[target_col].values[0]
                    features_only = random_row.drop(target_col, axis=1)
                    
                    st.write("**Données du trafic réseau extraites :**")
                    st.dataframe(features_only)
                    
                    # Prédiction
                    features_scaled = st.session_state['scaler'].transform(features_only)
                    prediction = st.session_state['model'].predict(features_scaled)[0]
                    proba = st.session_state['model'].predict_proba(features_scaled)
                    
                    st.markdown("### Résultat :")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Vraie étiquette : **{'DDoS (1)' if true_label == 1 else 'Normal (0)'}**")
                    with col2:
                        if prediction == 1:
                            st.error(f"Prédiction IA : **DDoS (1)** (Confiance: {proba[0][1]:.2%})")
                        else:
                            st.success(f"Prédiction IA : **Normal (0)** (Confiance: {proba[0][0]:.2%})")
                            
                    if true_label == prediction:
                        st.success("✅ L'IA a fait la bonne prédiction !")
                    else:
                        st.warning("❌ L'IA s'est trompée sur cette ligne.")

            # MODE 2 : MANUEL
            elif type_test == "Saisie Manuelle (Entrer vos propres valeurs)":
                st.subheader("Entrez les paramètres du trafic réseau :")
                user_inputs = {}
                cols = st.columns(3)
                
                # Création dynamique des inputs sans erreur de syntaxe
                for i, feature in enumerate(st.session_state['features']):
                    with cols[i % 3]:
                        # Utilise la moyenne de la colonne comme valeur par défaut
                        mean_val = float(st.session_state['df_full'][feature].mean())
                        user_inputs[feature] = st.number_input(f"{feature}", value=mean_val)

                if st.button("Analyser le trafic"):
                    input_df = pd.DataFrame([user_inputs])
                    input_scaled = st.session_state['scaler'].transform(input_df)
                    
                    prediction = st.session_state['model'].predict(input_scaled)[0]
                    proba = st.session_state['model'].predict_proba(input_scaled)

                    st.markdown("---")
                    if prediction == 1:
                        st.error(f"🚨 ALERTE : Trafic identifié comme **DDoS** (Confiance : {proba[0][1]:.2%})")
                    else:
                        st.success(f"✅ NORMAL : Trafic sain (Confiance : {proba[0][0]:.2%})")

else:
    st.info("👋 Bienvenue ! Veuillez charger votre fichier CSV dans le menu de gauche pour commencer.")