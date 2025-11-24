import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Ajout du chemin src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from eyes_detection import analyze_tadpole_microscope
from stats import calculate_significant_stats, detect_outliers_zscore
from report import generate_pdf_report

# --- CONFIGURATION ---
st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
st.title("🐸 Xenopus Morphometric Pipeline (Version M2 Finale)")

# --- MÉMOIRE ---
if 'df_resultats' not in st.session_state:
    st.session_state.df_resultats = None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")

# Default input path relative to the app
default_input_path = os.path.join(os.getcwd(), "data", "raw", "biométrie")
if not os.path.exists(default_input_path):
    default_input_path = os.getcwd()

dossier_input = st.sidebar.text_input("Dossier Images (Entrée) :", value=default_input_path)

# Results path (User requested specific default)
# Note: On Linux/Mac this default won't be writable, so we handle that gracefully if it fails.
default_results_path = r"C:\Users\User\Desktop\results\biométrie"
dossier_output = st.sidebar.text_input("Dossier Résultats (Sortie) :", value=default_results_path)

pixel_mm_ratio = st.sidebar.number_input("Calibration (mm/pixel)", value=0.0053, format="%.5f")
st.sidebar.info("Facteur correctif pour la queue transparente (basé sur la thèse).")
facteur_queue = st.sidebar.slider("Facteur Queue", 1.0, 4.0, 2.6, 0.1)

# --- CORPS DE L'APPLICATION ---
def run_app():
    # 1. BOUTON D'ANALYSE
    if st.sidebar.button("Lancer l'analyse 🚀", use_container_width=True):
        if not os.path.exists(dossier_input):
            st.error(f"Dossier d'entrée introuvable: {dossier_input}")
            return

        files = []
        for r, d, f in os.walk(dossier_input):
            for file in f:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    files.append(os.path.join(r, file))

        if not files:
            st.warning("Aucune image trouvée dans ce dossier.")
            return

        progress = st.progress(0)
        status = st.empty()
        res = []

        for i, path in enumerate(files):
            name = os.path.basename(path)
            status.text(f"Analyse : {name}")

            parts = path.split(os.sep)
            try:
                tank = parts[-2]
                cond = parts[-3]
            except:
                tank, cond = "Inc", "Inc"

            try:
                # Debug is False by default for batch in UI
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
                    "Statut": msg,
                    "Chemin_Complet": path
                })
            except Exception as e:
                res.append({
                    "Condition": cond, "Réplicat": tank, "Fichier": name,
                    "Statut": f"Erreur: {str(e)}",
                    "Dist_Yeux_mm": 0
                })

            progress.progress((i+1)/len(files))

        st.session_state.df_resultats = pd.DataFrame(res)
        status.text("✅ Terminé !")

    # 2. INTERFACE DE RÉSULTATS
    if st.session_state.df_resultats is not None:
        st.divider()
        st.header("1. Validation & Correction des Données")

        # --- OUTLIER DETECTION (IMPROVEMENT IDEA) ---
        if "Rapport" in st.session_state.df_resultats.columns:
            outliers = detect_outliers_zscore(st.session_state.df_resultats, "Rapport", threshold=3.0)
            if not outliers.empty:
                st.warning(f"⚠️ **Attention :** {len(outliers)} valeurs aberrantes détectées (Z-score > 3). Vérifiez les lignes ci-dessous.")
                st.dataframe(outliers[["Condition", "Fichier", "Rapport", "Z_Score"]].style.format({"Z_Score": "{:.2f}"}))
            else:
                st.success("✅ Aucune anomalie statistique majeure détectée (Z-score < 3).")

        st.info("Corrigez les valeurs aberrantes directement dans le tableau ci-dessous.")

        # TABLEAU ÉDITABLE
        df_final = st.data_editor(st.session_state.df_resultats, num_rows="dynamic", key="editor")

        # Filtre (exclusion des zéros pour les stats)
        if "Dist_Yeux_mm" in df_final.columns:
            df_clean = df_final[df_final["Dist_Yeux_mm"] > 0]
        else:
            df_clean = df_final

        # Ensure output directory exists
        try:
            os.makedirs(dossier_output, exist_ok=True)
            output_ready = True
        except Exception as e:
            st.error(f"Impossible de créer le dossier de sortie : {dossier_output}. ({e})")
            output_ready = False

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

        # 3. DASHBOARD SCIENTIFIQUE
        if not df_clean.empty and "Condition" in df_clean.columns and "Rapport" in df_clean.columns:
            st.divider()
            st.header("2. Analyse Statistique Automatisée")

            col_graph, col_stats = st.columns([2, 1])

            with col_graph:
                st.subheader("Distribution du Rapport Morphométrique")
                fig = px.box(df_clean, x="Condition", y="Rapport", color="Condition", points="all",
                             title="Comparaison Témoin vs Polluants")
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                st.subheader("Tests de Significativité 🧪")
                st.markdown("Comparaison statistique par rapport au **Témoin (T)**.")

                unique_conditions = df_clean["Condition"].unique()
                default_idx = 0
                if "T" in unique_conditions:
                    default_idx = list(unique_conditions).index("T")
                elif "Témoin" in unique_conditions:
                    default_idx = list(unique_conditions).index("Témoin")

                control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=default_idx)

                # --- CALCUL STATS ---
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

            # PDF EXPORT
            with col_export_pdf:
                if st.button("📄 Exporter Rapport PDF"):
                    if output_ready:
                        path_pdf = os.path.join(dossier_output, "Rapport_Analyse.pdf")
                        current_stats = df_stats if 'df_stats' in locals() else pd.DataFrame()
                        if generate_pdf_report(df_clean, current_stats, path_pdf):
                            st.success(f"Rapport PDF généré : {path_pdf}")
                        else:
                            st.error("Erreur lors de la génération du PDF.")

if __name__ == "__main__":
    run_app()
