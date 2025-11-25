import os

import pandas as pd
import streamlit as st

from src.egg_pipeline import run_egg_batch
from src.pipeline import run_tadpole_batch
from src.report import generate_pdf_report
from src.stats import calculate_significant_stats, detect_outliers_zscore


st.set_page_config(page_title="Xenopus Analysis Tool", layout="wide", page_icon="🐸")
st.title("🐸 Xenopus Morphometric Pipeline (Version M2 Finale)")


if "tadpole_results" not in st.session_state:
    st.session_state.tadpole_results = None
    st.session_state.tadpole_metadata = None

if "egg_results" not in st.session_state:
    st.session_state.egg_results = None
    st.session_state.egg_metadata = None


def run_app():
    st.sidebar.header("⚙️ Paramètres")

    mode_analyse = st.sidebar.radio(
        "Que voulez-vous analyser ?",
        ["Têtards (Morphométrie)", "Œufs (Fécondation)"],
    )

    default_input_path = os.path.join(os.getcwd(), "data", "raw", "biométrie")
    if not os.path.exists(default_input_path):
        default_input_path = os.getcwd()

    dossier_input = st.sidebar.text_input("Dossier Images (Entrée) :", value=default_input_path)
    dossier_output = st.sidebar.text_input(
        "Dossier Résultats (Sortie) :", value=os.path.join(os.getcwd(), "results")
    )

    pixel_mm_ratio = st.sidebar.number_input("Calibration (mm/pixel)", value=0.0053, format="%.5f")
    facteur_queue = st.sidebar.slider("Facteur Queue", 1.0, 4.0, 2.6, 0.1)

    if st.sidebar.button("Lancer l'analyse 🚀", use_container_width=True):
        try:
            if mode_analyse == "Têtards (Morphométrie)":
                df, meta = run_tadpole_batch(
                    dossier_input,
                    pixel_mm_ratio=pixel_mm_ratio,
                    tail_factor=facteur_queue,
                    output_dir=dossier_output,
                )
                st.session_state.tadpole_results = df
                st.session_state.tadpole_metadata = meta
            else:
                df, meta = run_egg_batch(dossier_input, output_dir=dossier_output)
                st.session_state.egg_results = df
                st.session_state.egg_metadata = meta
            st.success("✅ Analyse terminée")
        except Exception as exc:
            st.error(f"Erreur pendant l'analyse : {exc}")

    if mode_analyse == "Têtards (Morphométrie)":
        render_tadpole_results(dossier_output)
    else:
        render_egg_results()


def render_tadpole_results(dossier_output: str):
    df = st.session_state.tadpole_results
    meta = st.session_state.tadpole_metadata
    if df is None:
        return

    st.header("Résultats Têtards")
    if meta:
        st.caption(
            f"Images : {meta.get('n_images', 0)} | Erreurs : {meta.get('n_errors', 0)} | pixel/mm : {meta.get('pixel_mm_ratio')} | Facteur queue : {meta.get('tail_factor')}"
        )

    st.dataframe(df)

    if "ratio" in df.columns:
        outliers = detect_outliers_zscore(df, "ratio", threshold=3.0)
        if not outliers.empty:
            st.warning(f"⚠️ {len(outliers)} valeurs aberrantes détectées")
            st.dataframe(outliers[["condition", "frog_id", "ratio", "Z_Score"]])

    if "condition" in df.columns and "ratio" in df.columns and len(df) > 0:
        st.divider()
        st.subheader("Analyse Statistique")
        unique_conditions = df["condition"].unique()
        control_group = st.selectbox("Groupe témoin", unique_conditions, index=0)
        df_stats = calculate_significant_stats(df, "ratio", control_group=control_group)
        if not df_stats.empty:
            st.dataframe(df_stats, hide_index=True)

    if st.button("📄 Exporter Rapport PDF"):
        if generate_pdf_report(df, pd.DataFrame(), os.path.join(dossier_output, "Rapport_Analyse.pdf")):
            st.success("Rapport PDF généré")
        else:
            st.error("Erreur lors de la génération du PDF")


def render_egg_results():
    df = st.session_state.egg_results
    meta = st.session_state.egg_metadata
    if df is None:
        return

    st.header("Résultats Œufs")
    if meta:
        st.caption(f"Images : {meta.get('n_images', 0)} | Erreurs : {meta.get('n_errors', 0)}")
    st.dataframe(df)


if __name__ == "__main__":
    run_app()

