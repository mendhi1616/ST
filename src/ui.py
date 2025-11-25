import streamlit as st
import os

def setup_sidebar():
    """Configure la barre latérale avec tous les paramètres de l'application."""
    st.sidebar.header("⚙️ Paramètres")

    # Mode d'analyse
    st.sidebar.header("Mode d'Analyse")
    mode_analyse = st.sidebar.radio(
        "Que voulez-vous analyser ?",
        ["Têtards (Morphométrie)", "Œufs (Fécondation)"]
    )

    # Chemins des dossiers
    default_input_path = os.path.join(os.getcwd(), "data", "raw", "biométrie")
    if not os.path.exists(default_input_path):
        default_input_path = os.getcwd()

    dossier_input = st.sidebar.text_input(
        "Dossier Images (Entrée) :",
        value=default_input_path,
        help="Chemin vers le dossier contenant les images à analyser."
    )

    default_results_path = os.path.join(os.getcwd(), "results")
    dossier_output = st.sidebar.text_input(
        "Dossier Résultats (Sortie) :",
        value=default_results_path,
        help="Chemin où les rapports (Excel, PDF) seront sauvegardés."
    )

    # Upload de fichier unique
    st.sidebar.divider()
    st.sidebar.header("Analyse d'un Fichier Unique")
    uploaded_file = st.sidebar.file_uploader(
        "Ou chargez une image ici :",
        type=['.jpg', '.png', '.jpeg']
    )

    # Paramètres scientifiques
    st.sidebar.divider()
    st.sidebar.header("Paramètres Scientifiques")
    pixel_mm_ratio = st.sidebar.number_input(
        "Calibration (mm/pixel)",
        value=0.0053,
        format="%.5f",
        help="Facteur de conversion pour passer des pixels (image) aux millimètres (réel). Dépend du grossissement du microscope."
    )
    facteur_queue = st.sidebar.slider(
        "Facteur Queue", 1.0, 4.0, 2.6, 0.1,
        help="Facteur allométrique pour estimer la longueur totale du têtard (corps + queue transparente) à partir de la longueur du corps détectée."
    )

    params = {
        "mode_analyse": mode_analyse,
        "dossier_input": dossier_input,
        "dossier_output": dossier_output,
        "uploaded_file": uploaded_file,
        "pixel_mm_ratio": pixel_mm_ratio,
        "facteur_queue": facteur_queue
    }

    return params

def display_results(df_final, dossier_output):
    """Affiche l'interface des résultats, y compris les outliers, le tableau éditable et les options d'export."""
    st.divider()
    st.header("1. Validation & Correction des Données")

    # Détection des outliers
    if "Rapport" in df_final.columns:
        from stats import detect_outliers_zscore
        outliers = detect_outliers_zscore(df_final, "Rapport", threshold=3.0)
        if not outliers.empty:
            st.warning(f"⚠️ **Attention :** {len(outliers)} valeurs aberrantes détectées (Z-score > 3).")
            st.dataframe(outliers[["Condition", "Fichier", "Rapport", "Z_Score"]].style.format({"Z_Score": "{:.2f}"}))
        else:
            st.success("✅ Aucune anomalie statistique majeure détectée (Z-score < 3).")

    st.info("💡 Corrigez les valeurs si nécessaire directement dans le tableau ci-dessous.")

    # Tableau éditable
    df_edited = st.data_editor(df_final, num_rows="dynamic", key="editor")

    # Export
    col_export_excel, col_export_pdf = st.columns(2)
    output_ready = setup_output_directory(dossier_output)

    with col_export_excel:
        if st.button("💾 Sauvegarder Excel Final"):
            if output_ready:
                path_excel = os.path.join(dossier_output, "Resultats_Stage_Final.xlsx")
                df_to_save = df_edited.drop(columns=["Image_Annotée"], errors='ignore')
                df_to_save.to_excel(path_excel, index=False)
                st.success(f"Sauvegardé : {path_excel}")

    return df_edited, col_export_pdf

def setup_output_directory(path):
    """Crée le dossier de sortie s'il n'existe pas."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Impossible de créer le dossier de sortie : {path}. ({e})")
        return False
