import cv2
import numpy as np
import os
from .utils import read_image_with_unicode

def analyze_eggs(image_path, debug=False):
    if not os.path.exists(image_path):
        return None, 0, 0, "Fichier introuvable"

    img = read_image_with_unicode(image_path)
    if img is None:
        return None, 0, 0, "Image illisible"

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Prétraitement (Flou pour ignorer le bruit de fond)
    gray_blurred = cv2.medianBlur(gray, 5)

    # 2. Détection des Cercles (Hough Circle Transform)
    # Paramètres CRITIQUES à ajuster selon la taille de tes œufs sur l'image
    # minDist : Distance minimum entre deux œufs (pour ne pas compter le même deux fois)
    # param1 : Seuil Canny (plus haut = moins de bords détectés)
    # param2 : Seuil Accumulateur (plus bas = plus de cercles détectés, risque de faux positifs)
    # minRadius/maxRadius : Taille des œufs en pixels
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=40,       # <-- AJUSTE SI LES OEUFS SONT PROCHES OU ELOIGNÉS
        param1=50, 
        param2=30,        # <-- BAISSE SI TU NE VOIS PAS TOUS LES OEUFS
        minRadius=20,     # <-- TAILLE MINIMUM D'UN OEUF
        maxRadius=80      # <-- TAILLE MAXIMUM
    )

    count_fecondes = 0
    count_non_fecondes = 0
    status_msg = "Aucun œuf détecté"

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            
            # 3. Analyse du Contenu de l'Œuf (ROI)
            # On crée un masque pour analyser UNIQUEMENT l'intérieur du cercle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # On extrait les pixels de cet œuf
            # On érode un peu le masque pour ne pas prendre les bords de l'œuf (souvent sombres)
            mask = cv2.erode(mask, None, iterations=2)
            
            mean_val, std_dev = cv2.meanStdDev(gray, mask=mask)
            ecart_type = std_dev[0][0]
            
            # 4. Classification
            # SEUIL DE CONTRASTE :
            # Si l'écart-type est élevé (> 15), c'est qu'il y a du noir ET du blanc -> FÉCONDÉ
            # Si l'écart-type est faible (< 15), c'est uniforme -> NON FÉCONDÉ
            SEUIL_CONTRASTE = 15.0 
            
            if ecart_type > SEUIL_CONTRASTE:
                # FÉCONDÉ (Vert)
                color = (0, 255, 0) 
                label = "F"
                count_fecondes += 1
            else:
                # NON FÉCONDÉ (Rouge)
                color = (0, 0, 255) 
                label = "NF"
                count_non_fecondes += 1
            
            # Dessin
            cv2.circle(output_img, (center_x, center_y), radius, color, 2)
            cv2.circle(output_img, (center_x, center_y), 2, (0, 255, 255), 3) # Centre
            # Optionnel : Afficher le score std_dev pour debug
            if debug:
                cv2.putText(output_img, f"{int(ecart_type)}", (center_x-10, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        total = count_fecondes + count_non_fecondes
        if total > 0:
            percent = (count_fecondes / total) * 100
            status_msg = f"{total} Detectes | {percent:.1f}% Fecondes"
        
    return output_img, count_fecondes, count_non_fecondes, status_msg

# --- TEST RAPIDE ---
if __name__ == "__main__":
    # Change ça avec une de tes images d'œufs
    test_path = r"C:\Users\User\Desktop\Xenopus_Project\data\raw\biométrie\test_oeuf.jpg"
    
    # Recherche auto
    if not os.path.exists(test_path):
        for root, dirs, files in os.walk(r"C:\Users\User\Desktop\Xenopus_Project\data\raw"):
            for f in files:
                if f.endswith(".jpg") or f.endswith(".png"):
                    test_path = os.path.join(root, f)
                    break
            if os.path.exists(test_path): break
    
    print(f"Test sur : {test_path}")
    res_img, f, nf, msg = analyze_eggs(test_path, debug=True)
    print(f"Résultat : {msg}")
    print(f"Fécondés : {f}")
    print(f"Non Fécondés : {nf}")
    
    if res_img is not None:
        cv2.imwrite("test_oeufs_result.jpg", res_img)
        print("Image sauvegardée : test_oeufs_result.jpg")