import cv2
import numpy as np
import math
import os

def imread_windows_special(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

def analyze_tadpole_microscope(image_path, debug=False):
    # --- CHEMINS DE SORTIE ---
    # On définit où sauvegarder les images de debug
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    result_dir = os.path.join(base_dir, "data", "results")
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(image_path): return None, 0, 0, "Fichier introuvable"
    img = imread_windows_special(image_path)
    if img is None: return None, 0, 0, "Image illisible"
    
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. DÉTECTION DU CORPS
    _, mask_body = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)
    contours_body, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_body: return None, 0, 0, "Echec corps"
    c_body = max(contours_body, key=cv2.contourArea)
    
    # Mesure Longueur
    if len(c_body) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(c_body)
        body_length_px = max(MA, ma)
    else:
        rect = cv2.minAreaRect(c_body)
        body_length_px = max(rect[1])
        
    cv2.drawContours(output_img, [c_body], -1, (0, 255, 0), 2) # Corps en VERT

    # 2. DÉTECTION DES YEUX (STRATÉGIE ADAPTATIVE)
    body_only = cv2.bitwise_and(gray, gray, mask=mask_body)
    body_only[mask_body == 0] = 255 
    
    # Seuil standard
    _, mask_eyes = cv2.threshold(body_only, 65, 255, cv2.THRESH_BINARY_INV)
    contours_eyes, _ = cv2.findContours(mask_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # -- FILTRAGE --
    candidats_parfaits = [] # Ronds
    candidats_moyens = []   # Pas ronds mais bonne taille
    
    for c in contours_eyes:
        area = cv2.contourArea(c)
        if 10 < area < 600: # Filtre taille large
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Dessin de DEBUG (Orange = candidat détecté mais en attente)
            cv2.drawContours(output_img, [c], -1, (0, 165, 255), 1) 
            
            if circularity > 0.5: # Circularité > 0.5
                candidats_parfaits.append(c)
            else:
                candidats_moyens.append(c)

    # STRATÉGIE : On essaie d'abord avec les parfaits, sinon on pioche dans les moyens
    selection = []
    mode = ""
    
    # On trie par taille (du plus gros au plus petit)
    candidats_parfaits.sort(key=cv2.contourArea, reverse=True)
    candidats_moyens.sort(key=cv2.contourArea, reverse=True)

    # --- CORRECTION ICI --- (C'est là que j'avais fait la faute)
    if len(candidats_parfaits) >= 2:
        selection = candidats_parfaits[:2]
        mode = "Precise (Ronds)"
    elif len(candidats_parfaits) == 1 and len(candidats_moyens) >= 1:
        selection = [candidats_parfaits[0], candidats_moyens[0]]
        mode = "Mixte"
    elif len(candidats_moyens) >= 2:
        selection = candidats_moyens[:2]
        mode = "Fallback (Non-ronds)"
    
    eye_distance_px = 0
    status_msg = "Yeux HS"

    if len(selection) == 2:
        oeil_1 = selection[0]
        oeil_2 = selection[1]
        
        M1 = cv2.moments(oeil_1)
        M2 = cv2.moments(oeil_2)
        
        if M1["m00"] != 0 and M2["m00"] != 0:
            c1 = (int(M1["m10"]/M1["m00"]), int(M1["m01"]/M1["m00"]))
            c2 = (int(M2["m10"]/M2["m00"]), int(M2["m01"]/M2["m00"]))
            
            dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            
            # CHECK DE SÉCURITÉ : La distance doit être petite (< 25% de la longueur du corps)
            ratio_dist_corps = dist / body_length_px
            
            if ratio_dist_corps < 0.25: 
                eye_distance_px = dist
                # Dessin FINAL en ROUGE
                cv2.line(output_img, c1, c2, (0, 0, 255), 2)
                cv2.drawContours(output_img, selection, -1, (0, 0, 255), -1)
                status_msg = f"Succès ({mode})"
            else:
                status_msg = f"Rejet (Ecart trop grand: {int(dist)}px)"
                # On dessine en VIOLET pour montrer l'erreur
                cv2.line(output_img, c1, c2, (255, 0, 255), 2) 

    cv2.putText(output_img, f"L: {int(body_length_px)} | Yeux: {int(eye_distance_px)}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output_img, body_length_px, eye_distance_px, status_msg

# =======================================================
# ZONE DE LANCEMENT (Test)
# =======================================================
if __name__ == "__main__":
    target_folder = r"C:\Users\User\Desktop\Xenopus_Project\data\raw\biométrie"
    
    print(f"--- TEST SCRIPT V5.1 (CORRIGÉ) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_debug = os.path.join(base_dir, "data", "results", "debug_image.jpg")
    
    found_image = None
    for root, dirs, files in os.walk(target_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png')):
                found_image = os.path.join(root, filename)
                break
        if found_image: break
                
    if found_image:
        print(f"Analyse de : {found_image}")
        res_img, length, eyes, msg = analyze_tadpole_microscope(found_image)
        
        print(f"✅ RÉSULTAT : {msg}")
        print(f"   Longueur : {length:.2f}")
        print(f"   Dist. Yeux : {eyes:.2f}")
        
        if res_img is not None:
            cv2.imwrite(output_debug, res_img)
            print(f"📁 Image sauvegardée ici : {output_debug}")
            print("-> Va voir cette image pour comprendre la détection !")
    else:
        print("❌ Aucune image trouvée.")