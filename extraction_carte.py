import cv2
import numpy as np
import pytesseract

# NOTE: Sous Windows, vous devrez peut-être indiquer le chemin d'installation de Tesseract.
# Décommentez la ligne suivante et ajustez le chemin si nécessaire :
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extraire_informations(image_path):
    print(f"Traitement de l'image : {image_path}")
    
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")

    # Convertir l'image en espace colorimétrique HSV (idéal pour filtrer par couleur)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # ETAPE 1 : ISOLER LES CLÉS (TEXTE BLEU/CYAN)
    # ---------------------------------------------------------
    # Définir la plage de couleur pour le bleu/cyan de la carte
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Créer un masque qui ne garde que les pixels bleus
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Inverser le masque pour avoir le texte en noir sur fond blanc (préféré par l'OCR)
    img_cles = cv2.bitwise_not(mask_blue)

    # ---------------------------------------------------------
    # ETAPE 2 : ISOLER LES VALEURS (TEXTE NOIR)
    # ---------------------------------------------------------
    # Définir la plage pour la couleur noire (faible luminosité)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    
    # Créer un masque qui ne garde que les pixels noirs
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Inverser le masque
    img_valeurs = cv2.bitwise_not(mask_black)

    # ---------------------------------------------------------
    # ETAPE 3 : PASSER L'OCR (TESSERACT) AVEC LA LANGUE ALBANAISE
    # ---------------------------------------------------------
    # --psm 6 signifie "Assumer un bloc de texte uniforme"
    # lang='sqi' utilise le dictionnaire Albanais
    
    print("\n[!] Extraction des CLÉS en cours...")
    texte_cles = pytesseract.image_to_string(img_cles, lang='sqi+eng', config='--psm 6')
    
    print("[!] Extraction des VALEURS en cours...")
    texte_valeurs = pytesseract.image_to_string(img_valeurs, lang='sqi', config='--psm 6')

    return texte_cles, texte_valeurs

if __name__ == "__main__":
    # Remplacez "carte_albanie.jpg" par le nom réel de votre fichier image
    image_a_tester = "carte_albanie.jpg" 
    
    try:
        cles, valeurs = extraire_informations(image_a_tester)
        
        print("\n================ RÉSULTATS ================")
        print("\n--- CLÉS DÉTECTÉES (Extraites du Bleu) ---")
        print(cles)
        
        print("\n--- VALEURS DÉTECTÉES (Extraites du Noir) ---")
        print(valeurs)
        print("===========================================")
        
    except Exception as e:
        print(f"Erreur : {e}")
