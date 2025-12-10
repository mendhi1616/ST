import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from eyes_detection import analyze_tadpole_microscope

PIXEL_TO_MM = 0.0053
FACTEUR_QUEUE = 2.6

def process_dataset_batch(root_folder, output_folder=None):
    print(f"🚀 DÉMARRAGE DU TRAITEMENT PAR LOT")
    print(f"📂 Dossier : {root_folder}")
    if output_folder:
        print(f"💾 Sortie  : {output_folder}")
    print(f"📏 Calibration : 1 px = {PIXEL_TO_MM} mm")
    print(f"🧪 Correction Queue : x{FACTEUR_QUEUE}")
    print("-" * 60)

    data = []
    files_processed = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                full_path = os.path.join(root, file)
                files_processed += 1

                parts = full_path.split(os.sep)
                try:
                    replicat = parts[-2]     
                    condition = parts[-3]        
                    stage = parts[-4]             
                except:
                    replicat = "Inconnu"
                    condition = "Inconnu"
                    stage = "Inconnu"
                print(f"[{files_processed}] Traitement de {file}...", end="")


import os
import pandas as pd
import sys

# Ajout du chemin src pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from eyes_detection import analyze_tadpole_microscope
except ImportError as e:
    print(f"Erreur import: {e}")
    sys.exit(1)

# --- PARAMÈTRES DE CALIBRATION ---
PIXEL_TO_MM = 0.00263  # À vérifier selon ton microscope
FACTEUR_QUEUE = 3.5    # Estimation (Queue ≈ 3.5x le corps)

def run_batch_analysis(input_folder, output_csv="resultats_complets.csv"):
    data = []
    
    print(f"📂 Analyse du dossier : {input_folder}")
    
    # Parcours récursif (Conditions / Réplicats / Fichiers)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                full_path = os.path.join(root, file)
                
                # Déduction des métadonnées via l'arborescence
                parts = full_path.split(os.sep)
                # Ex: .../Condition/Replicat/Image.jpg
                if len(parts) >= 3:
                    condition = parts[-3]
                    replicat = parts[-2]
                else:
                    condition = "Inconnue"
                    replicat = "Inconnu"

                print(f"➡️ Traitement : {file}...", end="", flush=True)

                try:
                    # --- APPEL ALGO (6 VALEURS) ---
                    # On récupère aussi 'snout_px' (Nez) pour ne pas planter
                    _, len_px_corps, eyes_px, snout_px, status, orientation = analyze_tadpole_microscope(full_path, debug=False)
                    
                    # Conversions
                    corps_mm = len_px_corps * PIXEL_TO_MM
                    total_mm_estime = corps_mm * FACTEUR_QUEUE
                    eyes_mm = eyes_px * PIXEL_TO_MM
                    snout_mm = snout_px * PIXEL_TO_MM

                    # Calcul Ratio
                    if total_mm_estime > 0 and orientation == "dorsal":
                        ratio = eyes_mm / total_mm_estime
                    else:
                        ratio = 0

                    # Ajout aux données
                    data.append({
                        "Condition": condition,
                        "Réplicat": replicat,
                        "Fichier": file,
                        "Orientation": orientation,
                        "Longueur Corps (mm)": round(corps_mm, 3),
                        "Longueur Totale Est. (mm)": round(total_mm_estime, 3),
                        "Dist. Yeux (mm)": round(eyes_mm, 3),
                        "Dist. Nez (mm)": round(snout_mm, 3), # Nouvelle donnée utile !
                        "Rapport (Yeux/Total)": round(ratio, 4),
                        "Statut Algo": status,
                        "Chemin": full_path
                    })

                    if "Success" in status or "Succès" in status:
                        print(f" ✅ OK ({orientation})")
                    else:
                        print(f" ⚠️ {status}")

                except Exception as e:
                    print(f" ❌ ERREUR: {e}")
                    data.append({
                        "Fichier": file, 
                        "Statut Algo": f"Crash: {e}", 
                        "Orientation": "error",
                        "Chemin": full_path
                    })

    # Sauvegarde Excel
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False, sep=";", encoding="utf-8-sig")
        print(f"\n🎉 Analyse terminée ! Résultats : {output_csv}")
    else:
        print("\n⚠️ Aucune donnée n'a été générée.")

if __name__ == "__main__":
    # Dossier par défaut si non spécifié
    target_dir = r"C:\Users\User\Desktop\results" 
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        
    run_batch_analysis(target_dir)

    if data:
        if not output_folder:
            base_dir = os.path.dirname(os.path.dirname(root_folder))
            output_folder = os.path.join(base_dir, "results")
            if not os.path.exists(output_folder):
                output_folder = os.path.join(root_folder, "..", "Resultats_Analyse")

        try:
            os.makedirs(output_folder, exist_ok=True)
            excel_path = os.path.join(output_folder, "Resultats_Complets_Biometrie.xlsx")

            df = pd.DataFrame(data)

            cols = ["Condition", "Réplicat", "Fichier",
                    "Longueur Totale Est. (mm)", "Dist. Yeux (mm)", "Rapport (Yeux/Total)",
                    "Statut Algo", "Orientation", "Longueur Corps (mm)"]

            cols_existantes = [c for c in cols if c in df.columns]
            df = df[cols_existantes]

            df.to_excel(excel_path, index=False)
            print("-" * 60)
            print(f"✅ TERMINE ! {len(df)} lignes générées.")
            print(f"📊 Fichier Excel : {excel_path}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde Excel (Fichier ouvert ?) : {e}")

    else:
        print("❌ Aucune donnée à sauvegarder.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Analysis for Xenopus Morphometry")
    parser.add_argument("input_dir", nargs='?', default=os.path.join(os.getcwd(), "data", "raw", "biométrie"), help="Input directory containing images")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results")

    args = parser.parse_args()

    target = args.input_dir
    output_dir = args.output

    if os.path.exists(target):
        process_dataset_batch(target, output_dir)
    else:
        print(f"ERREUR : Le dossier n'existe pas : {target}")
