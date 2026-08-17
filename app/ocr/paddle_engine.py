import numpy as np
import cv2
from PIL import Image, ImageOps


class PaddleEngine:

    def __init__(self, use_gpu: bool = True):
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The PaddleOCR package is not installed. "
                "Install it with `pip install -r requirements.txt` or `pip install paddleocr`."
            ) from exc

        if use_gpu:
            # ── Mode GPU : inférence accélérée via CUDA ──
            self.ocr = PaddleOCR(
                lang="fr",
                use_gpu=True,
                gpu_id=0,
                show_log=False,
                det_limit_side_len=736,
                rec_batch_num=16,
                drop_score=0.5,
                use_dilation=False,
                det_db_score_mode="fast",
                use_angle_cls=False,
                enable_hpi=False,
                enable_mkldnn=False,
            )
        else:
            # ── Mode CPU : multiprocessing, chaque worker a son propre contexte ──
            self.ocr = PaddleOCR(
                lang="fr",
                use_gpu=False,
                det_limit_side_len=736,
                rec_batch_num=8,      # Batch réduit en CPU pour ne pas surcharger un seul worker
                use_angle_cls=False,
                enable_mkldnn=True,   # Accélération Intel MKL-DNN pour CPU
                cpu_threads=2,        # Limité pour laisser des cœurs aux autres workers
                show_log=False,
            )


    def _resize_image(self, img, max_side=736):
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img


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

        image = self._resize_image(image, max_side=960)

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