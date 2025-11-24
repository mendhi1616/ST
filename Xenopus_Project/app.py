import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from scipy import stats # La librairie pour les maths statistiques

# Ajout du chemin src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from eyes_detection import analyze_tadpole_microscope

# --- CONFIGURATION ---
st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
st.title("🐸 Xenopus Morphometric Pipeline (Version M2 Finale)")

# --- MÉMOIRE ---
if 'df_resultats' not in st.session_state:
    st.session_state.df_resultats = None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
dossier_input = st.sidebar.text_input("Dossier Images :", 
                                    value=r"C:\Users\User\Desktop\results\biométrie")
pixel_mm_ratio = st.sidebar.number_input("Calibration (mm/pixel)", value=0.0053, format="%.5f")
st.sidebar.info("Facteur correctif pour la queue transparente (basé sur la thèse).")
facteur_queue = st.sidebar.slider("Facteur Queue", 1.0, 4.0, 2.6, 0.1)

# --- FONCTION UTILITAIRE : STATISTIQUES ---
def calculer_stats_significatives(df, colonne_mesure, groupe_temoin="T"):
    """
    Calcule les p-values (Mann-Whitney) comparé au témoin.
    """
    resultats_stats = []
    
    # Vérifier si le témoin existe
    if groupe_temoin not in df["Condition"].unique():
        return pd.DataFrame() # Pas de témoin, pas de stats

    # Données du témoin
    data_temoin = df[df["Condition"] == groupe_temoin][colonne_mesure].dropna()
    
    for condition in df["Condition"].unique():
        if condition == groupe_temoin:
            continue # On ne compare pas le témoin avec lui-même
            
        data_cond = df[df["Condition"] == condition][colonne_mesure].dropna()
        
        if len(data_cond) > 1 and len(data_temoin) > 1:
            # Test de Mann-Whitney U (Non paramétrique, robuste)
            stat, p_value = stats.mannwhitneyu(data_temoin, data_cond, alternative='two-sided')
            
            # Étoiles de significativité
            if p_value < 0.001: stars = "***"
            elif p_value < 0.01: stars = "**"
            elif p_value < 0.05: stars = "*"
            else: stars = "ns"
            
            resultats_stats.append({
                "Comparaison": f"{groupe_temoin} vs {condition}",
                "Médiane Témoin": round(data_temoin.median(), 3),
                "Médiane Cond.": round(data_cond.median(), 3),
                "P-value": f"{p_value:.4f}",
                "Significativité": stars
            })
            
    return pd.DataFrame(resultats_stats)

# --- CORPS DE L'APPLICATION ---
def run_app():
    # 1. BOUTON D'ANALYSE
    if st.sidebar.button("Lancer l'analyse 🚀", use_container_width=True):
        if not os.path.exists(dossier_input):
            st.error("Dossier introuvable.")
            return

        files = []
        for r, d, f in os.walk(dossier_input):
            for file in f:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    files.append(os.path.join(r, file))
        
        if not files:
            st.warning("Aucune image.")
            return

        progress = st.progress(0)
        status = st.empty()
        res = []
        
        for i, path in enumerate(files):
            name = os.path.basename(path)
            status.text(f"Analyse : {name}")
            
            parts = path.split(os.sep)
            try: tank, cond = parts[-2], parts[-3]
            except: tank, cond = "Inc", "Inc"

            try:
                _, len_px, eyes_px, msg = analyze_tadpole_microscope(path, debug=False)
                
                corps_mm = len_px * pixel_mm_ratio
                total_mm = corps_mm * facteur_queue
                eyes_mm = eyes_px * pixel_mm_ratio
                
                ratio = (eyes_mm / total_mm) if total_mm > 0 else 0
                
                res.append({
                    "Condition": cond, "Réplicat": tank, "Fichier": name,
                    "Corps_mm": round(corps_mm, 3),
                    "Total_Estimé_mm": round(total_mm, 3),
                    "Dist_Yeux_mm": round(eyes_mm, 3),
                    "Rapport": round(ratio, 4),
                    "Statut": msg
                })
            except: pass
            progress.progress((i+1)/len(files))
            
        st.session_state.df_resultats = pd.DataFrame(res)
        status.text("✅ Terminé !")

    # 2. INTERFACE DE RÉSULTATS
    if st.session_state.df_resultats is not None:
        st.divider()
        st.header("1. Validation & Correction des Données")
        st.info("Corrigez les valeurs aberrantes directement dans le tableau ci-dessous.")
        
        # TABLEAU ÉDITABLE (Semaine 1)
        df_final = st.data_editor(st.session_state.df_resultats, num_rows="dynamic", key="editor")
        
        # Filtre (exclusion des zéros pour les stats)
        df_clean = df_final[df_final["Dist_Yeux_mm"] > 0]

        # EXPORT
        if st.button("💾 Sauvegarder Excel Final"):
            path = os.path.join(dossier_input, "Resultats_Stage_Final.xlsx")
            df_final.to_excel(path, index=False)
            st.success(f"Sauvegardé : {path}")

        # 3. DASHBOARD SCIENTIFIQUE (Semaine 2 & 4)
        if not df_clean.empty:
            st.divider()
            st.header("2. Analyse Statistique Automatisée")
            
            col_graph, col_stats = st.columns([2, 1])
            
            with col_graph:
                st.subheader("Distribution du Rapport Morphométrique")
                # Boxplot avec points pour voir la dispersion
                fig = px.box(df_clean, x="Condition", y="Rapport", color="Condition", points="all",
                             title="Comparaison Témoin vs Polluants")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_stats:
                st.subheader("Tests de Significativité 🧪")
                st.markdown("Comparaison statistique par rapport au **Témoin (T)**.")
                st.markdown("*(Test de Mann-Whitney, p < 0.05)*")
                
                # --- CALCUL AUTOMATIQUE DES STATS ---
                # On suppose que le témoin s'appelle 'T'
                df_stats = calculer_stats_significatives(df_clean, "Rapport", groupe_temoin="T")
                
                if not df_stats.empty:
                    st.dataframe(df_stats, hide_index=True)
                    
                    st.markdown("---")
                    st.write("**Interprétation :**")
                    for index, row in df_stats.iterrows():
                        if row["Significativité"] != "ns":
                            st.write(f"⚠️ La condition **{row['Comparaison'].split(' vs ')[1]}** induit une modification significative ({row['Significativité']}).")
                else:
                    st.warning("Impossible de calculer les stats (Vérifiez qu'il y a bien une condition nommée 'T').")

if __name__ == "__main__":
    run_app()