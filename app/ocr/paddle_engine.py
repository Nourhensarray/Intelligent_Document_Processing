import numpy as np
import cv2
from PIL import Image, ImageOps

from paddleocr import PaddleOCR


class PaddleEngine:

    def __init__(self):

        self.ocr = PaddleOCR(
            lang="fr",
            use_angle_cls=True,
            use_gpu=False
        )


    def extract(self, image_input, use_color_masking=False):
        """
        Lance l'OCR sur l'image.

        Args:
            image_input: Chemin du fichier ou image BGR déjà prétraitée.
            use_color_masking: Si True, sépare le texte bleu (clés) du texte noir (valeurs).

        Returns:
            Liste de dicts {"text", "confidence", "box", "type" (optionnel)}.
        """
        if isinstance(image_input, np.ndarray):
            image = image_input
        else:
            image = self.load_image(image_input)

        if use_color_masking:
            return self._extract_with_masks(image)

        result = self.ocr.ocr(image, cls=True)
        ocr_data = []
        if result:
            for line in result:
                if line is None: continue
                for item in line:
                    ocr_data.append({
                        "text": item[1][0],
                        "confidence": item[1][1],
                        "box": item[0],
                        "type": "unknown"
                    })
        return ocr_data

    def _extract_with_masks(self, image):
        """Applique les masques de couleur pour différencier clés et valeurs."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. Masque Bleu (Clés)
        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        img_cles = cv2.bitwise_not(mask_blue)
        img_cles = cv2.cvtColor(img_cles, cv2.COLOR_GRAY2BGR)

        # 2. Masque Noir (Valeurs)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 90])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        img_valeurs = cv2.bitwise_not(mask_black)
        img_valeurs = cv2.cvtColor(img_valeurs, cv2.COLOR_GRAY2BGR)

        ocr_data = []

        # Extraction des clés (texte bleu)
        res_cles = self.ocr.ocr(img_cles, cls=True)
        if res_cles:
            for line in res_cles:
                if line is None: continue
                for item in line:
                    ocr_data.append({
                        "text": item[1][0],
                        "confidence": item[1][1],
                        "box": item[0],
                        "type": "key"  # Marqué comme Clé
                    })

        # Extraction des valeurs (texte noir)
        res_val = self.ocr.ocr(img_valeurs, cls=True)
        if res_val:
            for line in res_val:
                if line is None: continue
                for item in line:
                    ocr_data.append({
                        "text": item[1][0],
                        "confidence": item[1][1],
                        "box": item[0],
                        "type": "value" # Marqué comme Valeur
                    })

        return ocr_data


    def load_image(self, image_path):
        """
        Charge l'image en appliquant la rotation EXIF si elle est
        présente.

        Les photos prises avec un téléphone stockent souvent les
        pixels "à plat" avec un tag EXIF Orientation qui indique
        comment l'afficher (ex: tourné de 90°/180°/270°). PaddleOCR
        (via cv2) ignore ce tag et lit les pixels bruts : si on lui
        passe directement le chemin du fichier, il peut donc "voir"
        une image tournée alors qu'elle s'affiche correctement dans
        n'importe quelle visionneuse d'images.

        Résultat concret observé : texte lu à l'envers/de travers,
        ordre de tri haut->bas complètement inversé, et un OCR qui
        produit du charabia sur une partie des zones de texte.

        En appliquant ImageOps.exif_transpose ici, on garantit que
        PaddleOCR reçoit toujours l'image dans la bonne orientation,
        peu importe le tag EXIF du fichier source.
        """

        pil_image = Image.open(image_path)

        pil_image = ImageOps.exif_transpose(pil_image)

        rgb_array = np.array(pil_image.convert("RGB"))

        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array