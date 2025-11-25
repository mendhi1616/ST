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

# Palette de couleurs fixe pour les conditions
CONDITION_COLOR_MAP = {
    "EH": "#1f77b4",
    "EL": "#ff7f0e",
    "EM": "#2ca02c",
    "Témoin": "#d62728",
    "temoin": "#d62728",
    "T": "#d62728",
}

# --- CONSTANTES ---
CONTROL_GROUP_ALIASES = {
    "temoin": "Témoin",
    "témoin": "Témoin",
    "t": "Témoin"
}

import re
from pathlib import Path

import re
from pathlib import Path

def parse_path_metadata(path: str):
    """
    Déduit (stage, condition, replicat) à partir du chemin.

    Gère :
    - .../biométrie/fécondation VI/EL/T1/image.jpg
    - .../biométrie/fécondation VI/EL/image.jpg
    - et évite de renvoyer 'biométrie' comme stade.
    """
    p = Path(path)
    parts = list(p.parts)

    stage = "Inconnu"
    condition = "Inc"
    replicat = "Inconnu"

    idx_bio = None
    for i, part in enumerate(parts):
        if part.lower() in ("biométrie", "biometrie"):
            idx_bio = i
            break

    if idx_bio is not None:
        if idx_bio + 1 < len(parts):
            stage = parts[idx_bio + 1]          
        if idx_bio + 2 < len(parts):
            condition_or_replicat = parts[idx_bio + 2]
            if re.fullmatch(r"[Tt]\d+", condition_or_replicat):
                replicat = condition_or_replicat
            else:
                condition = condition_or_replicat
        if idx_bio + 3 < len(parts):
            last = parts[idx_bio + 3]
            if re.fullmatch(r"[Tt]\d+", last):
                replicat = last
    else:
        if len(parts) >= 3:
            parent = parts[-2]
            grandparent = parts[-3]
            ggparent = parts[-4] if len(parts) >= 4 else "Inconnu"

            if re.fullmatch(r"[Tt]\d+", parent):
                replicat = parent
                condition = grandparent
                stage = ggparent
            else:
                condition = parent
                stage = grandparent

    return stage, condition, replicat



def add_significance_annotations(
    fig,
    df_analysis: pd.DataFrame,
    df_stats: pd.DataFrame,
    measure_col: str,
    control_group: str,
):
    """
    Ajoute des barres + étoiles de significativité sur un graphique Plotly
    à partir de df_stats (résultat de calculate_significant_stats).
    On suppose que chaque ligne correspond à "control_group vs autre condition".
    """

    if df_stats is None or df_stats.empty:
        return fig

    condition_col = None
    for candidate in ["Condition", "Condition_Test", "Condition Cond.", "Cond", "Condition_Comparée"]:
        if candidate in df_stats.columns:
            condition_col = candidate
            break

    if condition_col is None and "Comparaison" in df_stats.columns:
        def extract_condition(comp, control=control_group):
            if isinstance(comp, str) and "vs" in comp:
                a, b = [c.strip() for c in comp.split("vs")]
                if a == control:
                    return b
                elif b == control:
                    return a
            return None

        df_stats = df_stats.copy()
        df_stats["Condition"] = df_stats["Comparaison"].apply(extract_condition)
        condition_col = "Condition"

    if condition_col is None:
        return fig

    signif_col = None
    for candidate in ["Significativité", "Signif.", "Signif", "Significance", "Stars"]:
        if candidate in df_stats.columns:
            signif_col = candidate
            break

    if signif_col is None:
        return fig

    used_offsets = 0
    for _, row in df_stats.iterrows():
        cond = row[condition_col]
        stars = row[signif_col]

        if cond is None or pd.isna(cond):
            continue
        if isinstance(stars, str) and stars.lower() == "ns":
            continue

        y_control = df_analysis[df_analysis["Condition"] == control_group][measure_col]
        y_cond = df_analysis[df_analysis["Condition"] == cond][measure_col]

        if y_control.empty or y_cond.empty:
            continue

        y_max = max(y_control.max(), y_cond.max())
        y_bar = y_max * (1.05 + 0.08 * used_offsets)
        used_offsets += 1

        x0 = control_group
        x1 = cond

        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            xref="x",
            y0=y_bar,
            y1=y_bar,
            yref="y",
            line=dict(
                width=3,       
                color="white",  
            ),
        )

        fig.add_shape(
            type="line",
            x0=x0,
            x1=x0,
            xref="x",
            y0=y_bar,
            y1=y_bar * 0.995,
            yref="y",
            line=dict(
                width=3,
                color="white",
            ),
        )

        fig.add_shape(
            type="line",
            x0=x1,
            x1=x1,
            xref="x",
            y0=y_bar,
            y1=y_bar * 0.995,
            yref="y",
            line=dict(
                width=3,
                color="white",
            ),
        )

        fig.add_annotation(
        x=x0,
        y=y_bar,
        text=str(stars),
        showarrow=False,
        yshift=6,
        font=dict(
            color="white",  
            size=20,         
            family="Arial",  
        ),
    )

    return fig


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
    stage, cond, tank = parse_path_metadata(path)
    cond_normalized = CONTROL_GROUP_ALIASES.get(cond.lower(), cond)
    stage_normalized = stage.strip()

    try:
        processed_img, len_px, eyes_px, msg = analyze_tadpole_microscope(path, debug=False)

        corps_mm = len_px * params["pixel_mm_ratio"]
        total_mm = corps_mm * params["facteur_queue"]
        eyes_mm = eyes_px * params["pixel_mm_ratio"]
        ratio = (eyes_mm / total_mm) if total_mm > 0 else 0.0

        return {
            "Fécondation": stage_normalized,
            "Condition": cond_normalized,
            "Réplicat": tank,
            "Fichier": name,
            "Corps_mm": round(corps_mm, 3),
            "Total_Estimé_mm": round(total_mm, 3),
            "Dist_Yeux_mm": round(eyes_mm, 3),
            "Rapport": round(ratio, 4),
            "Statut": msg,
            "Chemin_Complet": path,
            "Image_Annotée": processed_img,
        }

    except Exception as e:
        return {
            "Fécondation": stage_normalized,
            "Condition": cond_normalized,
            "Réplicat": tank,
            "Fichier": name,
            "Corps_mm": 0.0,
            "Total_Estimé_mm": 0.0,
            "Dist_Yeux_mm": 0.0,
            "Rapport": 0.0,
            "Statut": f"Erreur: {str(e)}",
            "Chemin_Complet": path,
            "Image_Annotée": None,
        }


def process_egg_image(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    name = os.path.basename(path)
    stage, cond, tank = parse_path_metadata(path)

    cond_normalized = CONTROL_GROUP_ALIASES.get(cond.lower(), cond)
    stage_normalized = stage.strip()

    try:
        processed_img, fecondes, non_fecondes, msg = analyze_eggs(path, debug=False)

        total = fecondes + non_fecondes
        fertilization_rate = (fecondes / total) * 100 if total > 0 else 0.0

        return {
            "Fécondation": stage_normalized,
            "Condition": cond_normalized,
            "Réplicat": tank,
            "Fichier": name,
            "Oeufs_Fecondes": fecondes,
            "Oeufs_Non_Fecondes": non_fecondes,
            "Taux_Fecondation": round(fertilization_rate, 2),
            "Statut": msg,
            "Chemin_Complet": path,
            "Image_Annotée": processed_img,
        }

    except Exception as e:
        return {
            "Fécondation": stage_normalized,
            "Condition": cond_normalized,
            "Réplicat": tank,
            "Fichier": name,
            "Oeufs_Fecondes": 0,
            "Oeufs_Non_Fecondes": 0,
            "Taux_Fecondation": 0.0,
            "Statut": f"Erreur: {str(e)}",
            "Chemin_Complet": path,
            "Image_Annotée": None,
        }


def run_analysis(files: List[str], params: Dict[str, Any]):
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
    st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
    st.title("🐸 Xenopus Morphometric Mendhi APP")

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

        if params["mode_analyse"] == "Têtards (Morphométrie)":
            df_clean = df_final[df_final["Dist_Yeux_mm"] > 0] if "Dist_Yeux_mm" in df_final.columns else df_final
            if not df_clean.empty and "Condition" in df_clean.columns and "Rapport" in df_clean.columns:
                st.divider()
                st.header("3. Analyse Statistique Automatisée")
                stages = ["Toutes"]
                if "Fécondation" in df_clean.columns:
                    stages += sorted(df_clean["Fécondation"].unique())

                col_graph, col_stats = st.columns([2, 1])

                with col_graph:
                    st.subheader("Distribution du Rapport Morphométrique")
                    selected_stage = st.selectbox(
                        "Fécondation / Stade analysé :",
                        stages,
                        index=0,
                    )

                if selected_stage != "Toutes" and "Fécondation" in df_clean.columns:
                    df_analysis = df_clean[df_clean["Fécondation"] == selected_stage].copy()
                else:
                    df_analysis = df_clean.copy()

                unique_conditions = sorted(df_analysis["Condition"].unique())
                control_index = 0
                if "Témoin" in unique_conditions:
                    control_index = unique_conditions.index("Témoin")

                with col_stats:
                    st.subheader("Tests de Significativité 🧪")
                    control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=control_index)

                df_stats = calculate_significant_stats(df_analysis, "Rapport", control_group=control_group)

                fig = px.box(
                    df_analysis,
                    x="Condition",
                    y="Rapport",
                    color="Condition",
                    points="all",
                    title="Comparaison Témoin vs Polluants",
                    color_discrete_map=CONDITION_COLOR_MAP,
                )

                fig = add_significance_annotations(
                    fig,
                    df_analysis=df_analysis,
                    df_stats=df_stats,
                    measure_col="Rapport",
                    control_group=control_group,
                )

                with col_graph:
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    if df_stats is not None and not df_stats.empty:
                        st.dataframe(df_stats, hide_index=True)

                with col_export_pdf:
                    if st.button("📄 Exporter Rapport PDF"):
                        path_pdf = os.path.join(params["dossier_output"], "Rapport_Analyse.pdf")
                        if generate_pdf_report(df_analysis, df_stats, path_pdf):
                            st.success(f"Rapport PDF généré : {path_pdf}")
                        else:
                            st.error("Erreur lors de la génération du PDF.")

        elif params["mode_analyse"] == "Œufs (Fécondation)":
            df_clean = df_final[df_final["Oeufs_Fecondes"] > 0] if "Oeufs_Fecondes" in df_final.columns else df_final
            if not df_clean.empty and "Condition" in df_clean.columns and "Taux_Fecondation" in df_clean.columns:
                st.divider()
                st.header("3. Analyse Statistique Automatisée")

                stages = ["Toutes"]
                if "Fécondation" in df_clean.columns:
                    stages += sorted(df_clean["Fécondation"].unique())

                col_graph, col_stats = st.columns([2, 1])

                with col_graph:
                    st.subheader("Distribution du Taux de Fécondation")
                    selected_stage = st.selectbox(
                        "Fécondation / Stade analysé :",
                        stages,
                        index=0,
                    )

                if selected_stage != "Toutes" and "Fécondation" in df_clean.columns:
                    df_analysis = df_clean[df_clean["Fécondation"] == selected_stage].copy()
                else:
                    df_analysis = df_clean.copy()

                unique_conditions = sorted(df_analysis["Condition"].unique())
                control_index = 0
                if "Témoin" in unique_conditions:
                    control_index = unique_conditions.index("Témoin")

                with col_stats:
                    st.subheader("Tests de Significativité 🧪")
                    control_group = st.selectbox("Groupe Témoin :", unique_conditions, index=control_index)

                df_stats = calculate_significant_stats(df_analysis, "Taux_Fecondation", control_group=control_group)

                fig = px.bar(
                    df_analysis,
                    x="Condition",
                    y="Taux_Fecondation",
                    color="Condition",
                    title="Comparaison du Taux de Fécondation",
                    color_discrete_map=CONDITION_COLOR_MAP,
                )

                fig = add_significance_annotations(
                    fig,
                    df_analysis=df_analysis,
                    df_stats=df_stats,
                    measure_col="Taux_Fecondation",
                    control_group=control_group,
                )

                with col_graph:
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    if df_stats is not None and not df_stats.empty:
                        st.dataframe(df_stats, hide_index=True)


    
if __name__ == "__main__":
    main()
