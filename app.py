import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Ajout du chemin src pour importer les modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import des modules personnalisés
from eyes_detection import analyze_tadpole_microscope
from egg_counting import analyze_eggs
from stats import calculate_significant_stats, detect_outliers_zscore
from report import generate_pdf_report

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
st.title("🐸 Xenopus Morphometric Pipeline (Version M2 Finale)")

# --- INITIALISATION DE LA MÉMOIRE (SESSION STATE) ---
if 'df_resultats' not in st.session_state:
    st.session_state.df_resultats = None

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")

# 1. Mode d'analyse
st.sidebar.header("Mode d'Analyse")
mode_analyse = st.sidebar.radio(
    "Que voulez-vous analyser ?", 
    ["Têtards (Morphométrie)", "Œufs (Fécondation)"]
)

# 2. Chemins des dossiers
# Chemin par défaut relatif à l'application
default_input_path = os.path.join(os.getcwd(), "data", "raw", "biométrie")
if not os.path.exists(default_input_path):
    default_input_path = os.getcwd()

dossier_input = st.sidebar.text_input("Dossier Images (Entrée) :", value=default_input_path)

# Chemin de sortie (Windows par défaut, modifiable)
default_results_path = r"C:\Users\User\Desktop\results\biométrie"
dossier_output = st.sidebar.text_input("Dossier Résultats (Sortie) :", value=default_results_path)

# 3. Paramètres scientifiques
pixel_mm_ratio = st.sidebar.number_input("Calibration (mm/pixel)", value=0.0053, format="%.5f")
st.sidebar.info("Facteur correctif pour la queue transparente (basé sur la thèse).")
facteur_queue = st.sidebar.slider("Facteur Queue", 1.0, 4.0, 2.6, 0.1)

# --- FONCTION PRINCIPALE ---
def run_app():
    # ==========================================
    # 1. LANCEMENT DE L'ANALYSE
    # ==========================================
    if st.sidebar.button("Lancer l'analyse 🚀", use_container_width=True):
        # Vérification du dossier d'entrée
        if not os.path.exists(dossier_input):
            st.error(f"Dossier d'entrée introuvable: {dossier_input}")
            return

        # Récupération des fichiers images
        files = []
        for r, d, f in os.walk(dossier_input):
            for file in f:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    files.append(os.path.join(r, file))

        if not files:
            st.warning("Aucune image trouvée dans ce dossier.")
            return

        # Initialisation de la barre de progression
        progress = st.progress(0)
        status = st.empty()
        res = []

        # Boucle de traitement
        for i, path in enumerate(files):
            name = os.path.basename(path)
            status.text(f"Analyse : {name}")

            # Extraction des métadonnées (Condition/Tank) depuis le chemin
            parts = path.split(os.sep)
            try:
                tank = parts[-2]
                cond = parts[-3]
            except:
                tank, cond = "Inc", "Inc"

            try:
                # Analyse de l'image (Appel au Backend)
                # debug=False pour accélérer le traitement en lot
                _, len_px, eyes_px, msg = analyze_tadpole_microscope(path, debug=False)

                # Conversions Pixel -> Millimètre
                corps_mm = len_px * pixel_mm_ratio
                total_mm = corps_mm * facteur_queue
                eyes_mm = eyes_px * pixel_mm_ratio

                # Calcul du Ratio
                ratio = (eyes_mm / total_mm) if total_mm > 0 else 0

                res.append({
                    "Condition": cond, 
                    "Réplicat": tank, 
                    "Fichier": name,
                    "Corps_mm": round(corps_mm, 3),
                    "Total_Estimé_mm": round(total_mm, 3),
                    "Dist_Yeux_mm": round(eyes_mm, 3),
                    "Rapport": round(ratio, 4),
                    "Statut": msg,
                    "Chemin_Complet": path
                })
            except Exception as e:
                res.append({
                    "Condition": cond, 
                    "Réplicat": tank, 
                    "Fichier": name,
                    "Statut": f"Erreur: {str(e)}",
                    "Dist_Yeux_mm": 0
                })

            # Mise à jour progression
            progress.progress((i+1)/len(files))

        # Stockage des résultats en mémoire
        st.session_state.df_resultats = pd.DataFrame(res)
        status.text("✅ Terminé !")

    # ==========================================
    # 2. INTERFACE DE RÉSULTATS & VALIDATION
    # ==========================================
    if st.session_state.df_resultats is not None:
        st.divider()
        st.header("1. Validation & Correction des Données")

        # --- DÉTECTION DES OUTLIERS (Valeurs Aberrantes) ---
        if "Rapport" in st.session_state.df_resultats.columns:
            outliers = detect_outliers_zscore(st.session_state.df_resultats, "Rapport", threshold=3.0)
            if not outliers.empty:
                st.warning(f"⚠️ **Attention :** {len(outliers)} valeurs aberrantes détectées (Z-score > 3). Vérifiez les lignes ci-dessous.")
                # Affiche un petit tableau des erreurs probables
                st.dataframe(outliers[["Condition", "Fichier", "Rapport", "Z_Score"]].style.format({"Z_Score": "{:.2f}"}))
            else:
                st.success("✅ Aucune anomalie statistique majeure détectée (Z-score < 3).")

        st.info("💡 Corrigez les valeurs aberrantes directement dans le tableau ci-dessous.")

        # --- TABLEAU ÉDITABLE (Human-in-the-loop) ---
        df_final = st.data_editor(st.session_state.df_resultats, num_rows="dynamic", key="editor")

        # Filtre : On exclut les zéros (échecs de détection) pour les stats
        if "Dist_Yeux_mm" in df_final.columns:
            df_clean = df_final[df_final["Dist_Yeux_mm"] > 0]
        else:
            df_clean = df_final

        # Préparation du dossier de sortie
        try:
            os.makedirs(dossier_output, exist_ok=True)
            output_ready = True
        except Exception as e:
            st.error(f"Impossible de créer le dossier de sortie : {dossier_output}. ({e})")
            output_ready = False

        # Boutons d'export
        col_export_excel, col_export_pdf = st.columns(2)

        with col_export_excel:
            if st.button("💾 Sauvegarder Excel Final"):
                if output_ready:
                    path_excel = os.path.join(dossier_output, "Resultats_Stage_Final.xlsx")
                    try:
                        df_final.to_excel(path_excel, index=False)
                        st.success(f"Sauvegardé : {path_excel}")
                    except Exception as e:
                        st.error(f"Erreur sauvegarde : {e}")

        # ==========================================
        # 3. DASHBOARD SCIENTIFIQUE (STATS)
        # ==========================================
        if not df_clean.empty and "Condition" in df_clean.columns and "Rapport" in df_clean.columns:
            st.divider()
            st.header("2. Analyse Statistique Automatisée")

            col_graph, col_stats = st.columns([2, 1])

            # GRAPHIQUE (Boxplot)
            with col_graph:
                st.subheader("Distribution du Rapport Morphométrique")
                fig = px.box(df_clean, x="Condition", y="Rapport", color="Condition", points="all",
                             title="Comparaison Témoin vs Polluants")
                st.plotly_chart(fig, use_container_width=True)

            # STATISTIQUES (Tableau P-values)
            with col_stats:
                st.subheader("Tests de Significativité 🧪")
                st.markdown("Comparaison statistique par rapport au **Témoin (T)**.")

                # Détection automatique du groupe Témoin
                unique_conditions = df_clean["Condition"].unique()
                default_idx = 0
                if "T" in unique_conditions:
                    default_idx = list(unique_conditions).index("T")
                elif "Témoin" in unique_conditions:
                    default_idx = list(unique_conditions).index("Témoin")

                control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=default_idx)

                # Calcul des stats via src/stats.py
                df_stats = calculate_significant_stats(df_clean, "Rapport", control_group=control_group)

                if not df_stats.empty:
                    st.dataframe(df_stats, hide_index=True)

                    st.markdown("---")
                    st.write("**Interprétation :**")
                    for index, row in df_stats.iterrows():
                        if row["Significativité"] != "ns":
                            st.write(f"⚠️ La condition **{row['Comparaison'].split(' vs ')[1]}** induit une modification significative ({row['Significativité']}).")
                else:
                    st.warning("Pas assez de données pour les statistiques.")

            # EXPORT PDF (Si les stats sont prêtes)
            with col_export_pdf:
                if st.button("📄 Exporter Rapport PDF"):
                    if output_ready:
                        path_pdf = os.path.join(dossier_output, "Rapport_Analyse.pdf")
                        current_stats = df_stats if 'df_stats' in locals() else pd.DataFrame()
                        
                        # Génération du PDF via src/report.py
                        if generate_pdf_report(df_clean, current_stats, path_pdf):
                            st.success(f"Rapport PDF généré : {path_pdf}")
                        else:
                            st.error("Erreur lors de la génération du PDF.")

if __name__ == "__main__":
    run_app()