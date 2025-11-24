import os
import pandas as pd
import sys

# On s'assure que Python trouve le fichier eyes_detection.py qui est dans le même dossier
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from eyes_detection import analyze_tadpole_microscope

# ==========================================
# CONFIGURATION SCIENTIFIQUE
# ==========================================
# 1. Calibration Microscope (à ajuster selon ton labo)
PIXEL_TO_MM = 0.0053  

# 2. Facteur de Correction Morphologique
# Le logiciel détecte le corps (Tête + Abdomen) de manière très fiable.
# La queue étant transparente, on l'estime par proportionnalité.
# Hypothèse : La longueur totale = 2.6 x Longueur Corps (pour Xenopus st. 45)
FACTEUR_QUEUE = 2.6 

def process_dataset_batch(root_folder):
    print(f"🚀 DÉMARRAGE DU TRAITEMENT PAR LOT")
    print(f"📂 Dossier : {root_folder}")
    print(f"📏 Calibration : 1 px = {PIXEL_TO_MM} mm")
    print(f"🧪 Correction Queue : x{FACTEUR_QUEUE}")
    print("-" * 60)
    
    data = []
    files_processed = 0
    
    # Parcours récursif des dossiers
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                full_path = os.path.join(root, file)
                files_processed += 1
                
                # --- 1. MÉTADONNÉES (CONDITION / TANK) ---
                # On découpe le chemin pour trouver les infos
                # Ex: .../biométrie/EH/T1/MC12001.JPG
                parts = full_path.split(os.sep)
                try:
                    replicat = parts[-2]  # Ex: T1
                    condition = parts[-3] # Ex: EH
                except:
                    replicat = "Inconnu"
                    condition = "Inconnu"
                
                print(f"[{files_processed}] Traitement de {file}...", end="")
                
                # --- 2. ANALYSE D'IMAGE (Moteur V5.1) ---
                try:
                    # On appelle la fonction de détection
                    _, len_px_corps, eyes_px, status = analyze_tadpole_microscope(full_path, debug=False)
                    
                    # --- 3. CALCULS BIOLOGIQUES ---
                    # A. Conversion en mm (Corps seul)
                    corps_mm = len_px_corps * PIXEL_TO_MM
                    
                    # B. Estimation de la Longueur Totale (avec Queue)
                    total_mm_estime = corps_mm * FACTEUR_QUEUE
                    
                    # C. Distance Yeux
                    eyes_mm = eyes_px * PIXEL_TO_MM
                    
                    # D. Le Rapport (Ratio)
                    # On utilise la longueur ESTIMÉE au dénominateur pour retrouver le ratio ~0.18
                    if total_mm_estime > 0:
                        ratio = eyes_mm / total_mm_estime
                    else:
                        ratio = 0
                    
                    # Stockage
                    data.append({
                        "Condition": condition,
                        "Réplicat": replicat,
                        "Fichier": file,
                        "Longueur Corps (mm)": round(corps_mm, 3),
                        "Longueur Totale Est. (mm)": round(total_mm_estime, 3),
                        "Dist. Yeux (mm)": round(eyes_mm, 3),
                        "Rapport (Yeux/Total)": round(ratio, 4),
                        "Statut Algo": status,
                        "Chemin": full_path
                    })
                    
                    if "Succès" in status:
                        print(f" OK (Rapport: {ratio:.3f})")
                    else:
                        print(f" ⚠️ {status}")
                        
                except Exception as e:
                    print(f" ERREUR: {e}")
                    data.append({"Fichier": file, "Statut Algo": f"Crash: {e}"})

    # --- 4. EXPORT EXCEL ---
    if data:
        # Chemin de sortie : Dans le dossier parent de "data/raw" -> "data/results"
        base_dir = os.path.dirname(os.path.dirname(root_folder)) # Remonte de 'biométrie' vers 'raw' vers 'data'
        output_folder = os.path.join(base_dir, "results") # Ça va créer data/results
        
        # Si le calcul de chemin est compliqué, on sauvegarde simplement à côté du dossier images
        if not os.path.exists(output_folder):
            output_folder = os.path.join(root_folder, "..", "Resultats_Analyse")
            
        os.makedirs(output_folder, exist_ok=True)
        excel_path = os.path.join(output_folder, "Resultats_Complets_Biometrie.xlsx")
        
        df = pd.DataFrame(data)
        
        # Réorganiser les colonnes pour faire propre
        cols = ["Condition", "Réplicat", "Fichier", 
                "Longueur Totale Est. (mm)", "Dist. Yeux (mm)", "Rapport (Yeux/Total)", 
                "Statut Algo", "Longueur Corps (mm)"]
        
        # On filtre pour ne garder que les colonnes qui existent
        cols_existantes = [c for c in cols if c in df.columns]
        df = df[cols_existantes]
        
        try:
            df.to_excel(excel_path, index=False)
            print("-" * 60)
            print(f"✅ TERMINE ! {len(df)} lignes générées.")
            print(f"📊 Fichier Excel : {excel_path}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde Excel (Fichier ouvert ?) : {e}")
            
    else:
        print("❌ Aucune donnée à sauvegarder.")

# ==========================================
# LANCEMENT DIRECT
# ==========================================
if __name__ == "__main__":
    # Mets ici le chemin de ton dossier images
    target = r"C:\Users\User\Desktop\Xenopus_Project\data\raw\biométrie"
    
    if os.path.exists(target):
        process_dataset_batch(target)
    else:
        print(f"ERREUR : Le dossier n'existe pas : {target}")