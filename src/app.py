import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
from typing import List, Dict, Any

# Import des modules personnalisés
from eyes_detection import analyze_tadpole_microscope
from egg_counting import analyze_eggs
from stats import calculate_significant_stats
from report import generate_pdf_report
from ui import setup_sidebar, display_results

# --- CONSTANTES ---
CONTROL_GROUP_ALIASES = {
    "temoin": "Témoin",
    "témoin": "Témoin",
    "t": "Témoin"
}

def get_image_files(input_path: str) -> List[str]:
    """Récupère la liste des chemins de fichiers image à partir d'un dossier."""
    files = []
    for r, d, f in os.walk(input_path):
        for file in f:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                files.append(os.path.join(r, file))
    return files

def process_tadpole_image(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Traite une seule image de têtard et retourne un dictionnaire de résultats."""
    name = os.path.basename(path)

    parts = path.split(os.sep)
    try:
        tank = parts[-2]
        cond = parts[-3]
    except IndexError:
        tank, cond = "Inc", "Inc"

    cond_normalized = CONTROL_GROUP_ALIASES.get(cond.lower(), cond)

    try:
        processed_img, len_px, eyes_px, msg = analyze_tadpole_microscope(path, debug=False)

        corps_mm = len_px * params["pixel_mm_ratio"]
        total_mm = corps_mm * params["facteur_queue"]
        eyes_mm = eyes_px * params["pixel_mm_ratio"]
        ratio = (eyes_mm / total_mm) if total_mm > 0 else 0

        return {
            "Condition": cond_normalized, "Réplicat": tank, "Fichier": name,
            "Corps_mm": round(corps_mm, 3), "Total_Estimé_mm": round(total_mm, 3),
            "Dist_Yeux_mm": round(eyes_mm, 3), "Rapport": round(ratio, 4),
            "Statut": msg, "Chemin_Complet": path, "Image_Annotée": processed_img
        }
    except Exception as e:
        return {
            "Condition": cond_normalized, "Réplicat": tank, "Fichier": name,
            "Statut": f"Erreur: {str(e)}", "Dist_Yeux_mm": 0, "Image_Annotée": None
        }

def process_egg_image(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Traite une seule image d'œuf et retourne un dictionnaire de résultats."""
    name = os.path.basename(path)

    parts = path.split(os.sep)
    try:
        tank = parts[-2]
        cond = parts[-3]
    except IndexError:
        tank, cond = "Inc", "Inc"

    cond_normalized = CONTROL_GROUP_ALIASES.get(cond.lower(), cond)

    try:
        processed_img, fecondes, non_fecondes, msg = analyze_eggs(path, debug=False)

        total = fecondes + non_fecondes
        fertilization_rate = (fecondes / total) * 100 if total > 0 else 0

        return {
            "Condition": cond_normalized, "Réplicat": tank, "Fichier": name,
            "Oeufs_Fecondes": fecondes, "Oeufs_Non_Fecondes": non_fecondes,
            "Taux_Fecondation": round(fertilization_rate, 2),
            "Statut": msg, "Chemin_Complet": path, "Image_Annotée": processed_img
        }
    except Exception as e:
        return {
            "Condition": cond_normalized, "Réplicat": tank, "Fichier": name,
            "Statut": f"Erreur: {str(e)}", "Oeufs_Fecondes": 0, "Image_Annotée": None
        }

def run_analysis(files: List[str], params: Dict[str, Any]):
    """Exécute l'analyse sur une liste de fichiers et met à jour le session state."""
    progress = st.progress(0)
    status = st.empty()
    results = []

    analysis_function = process_tadpole_image
    if params["mode_analyse"] == "Œufs (Fécondation)":
        analysis_function = process_egg_image

    for i, path in enumerate(files):
        name = os.path.basename(path)
        status.text(f"Analyse : {name}")
        results.append(analysis_function(path, params))
        progress.progress((i + 1) / len(files))

    st.session_state.df_resultats = pd.DataFrame(results)
    status.text("✅ Terminé !")

def main():
    """Fonction principale de l'application Streamlit."""
    st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
    st.title("🐸 Xenopus Morphometric Pipeline (Version M2 Finale)")

    if 'df_resultats' not in st.session_state:
        st.session_state.df_resultats = None

    params = setup_sidebar()

    if st.sidebar.button("Lancer l'analyse 🚀", use_container_width=True) or params["uploaded_file"]:
        if params["uploaded_file"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(params["uploaded_file"].getbuffer())
                files = [tmp.name]
        else:
            if not os.path.exists(params["dossier_input"]):
                st.error(f"Dossier d'entrée introuvable: {params['dossier_input']}")
                return
            files = get_image_files(params["dossier_input"])

        if not files:
            st.error("**Aucune image (.jpg, .png, .jpeg) trouvée.**")
            return

        run_analysis(files, params)

    if st.session_state.df_resultats is not None:
        df_final, col_export_pdf = display_results(st.session_state.df_resultats, params["dossier_output"], params["mode_analyse"])

        # Le reste de l'interface (graphiques, stats, etc.) reste ici pour le moment
        if params["mode_analyse"] == "Têtards (Morphométrie)":
            df_clean = df_final[df_final["Dist_Yeux_mm"] > 0] if "Dist_Yeux_mm" in df_final.columns else df_final
            if not df_clean.empty and "Condition" in df_clean.columns and "Rapport" in df_clean.columns:
                st.divider()
                st.header("3. Analyse Statistique Automatisée")

                col_graph, col_stats = st.columns([2, 1])

                with col_graph:
                    st.subheader("Distribution du Rapport Morphométrique")
                    fig = px.box(df_clean, x="Condition", y="Rapport", color="Condition", points="all", title="Comparaison Témoin vs Polluants")
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    st.subheader("Tests de Significativité 🧪")
                    unique_conditions = sorted(df_clean["Condition"].unique())

                    control_index = 0
                    if "Témoin" in unique_conditions:
                        control_index = unique_conditions.index("Témoin")

                    control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=control_index)

                    df_stats = calculate_significant_stats(df_clean, "Rapport", control_group=control_group)
                    if not df_stats.empty:
                        st.dataframe(df_stats, hide_index=True)

                with col_export_pdf:
                    if st.button("📄 Exporter Rapport PDF"):
                        path_pdf = os.path.join(params["dossier_output"], "Rapport_Analyse.pdf")
                        if generate_pdf_report(df_clean, df_stats, path_pdf):
                            st.success(f"Rapport PDF généré : {path_pdf}")
                        else:
                            st.error("Erreur lors de la génération du PDF.")

        elif params["mode_analyse"] == "Œufs (Fécondation)":
            df_clean = df_final[df_final["Oeufs_Fecondes"] > 0] if "Oeufs_Fecondes" in df_final.columns else df_final
            if not df_clean.empty and "Condition" in df_clean.columns and "Taux_Fecondation" in df_clean.columns:
                st.divider()
                st.header("3. Analyse Statistique Automatisée")

                col_graph, col_stats = st.columns([2, 1])

                with col_graph:
                    st.subheader("Distribution du Taux de Fécondation")
                    fig = px.bar(df_clean, x="Condition", y="Taux_Fecondation", color="Condition", title="Comparaison du Taux de Fécondation")
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    st.subheader("Tests de Significativité 🧪")
                    unique_conditions = sorted(df_clean["Condition"].unique())

                    control_index = 0
                    if "Témoin" in unique_conditions:
                        control_index = unique_conditions.index("Témoin")

                    control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=control_index)

                    df_stats = calculate_significant_stats(df_clean, "Taux_Fecondation", control_group=control_group)
                    if not df_stats.empty:
                        st.dataframe(df_stats, hide_index=True)


if __name__ == "__main__":
    main()
