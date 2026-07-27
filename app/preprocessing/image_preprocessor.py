import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


class ImagePreprocessor:
    """
    Prétraitement adaptatif d'une image de document après validation qualité.

    Étapes appliquées conditionnellement selon les métriques de qualité :
    1. Redimensionnement  – si la résolution est insuffisante pour l'OCR
    2. Débruitage         – toujours (léger, non-destructif)
    3. CLAHE              – si le contraste local est faible
    4. Unsharp mask       – si la netteté des caractères est insuffisante
    5. Deskew             – correction de la rotation (Hough lines)
    6. Sauvegarde debug   – dans outputs/preprocessed/ (optionnel)
    """

    # Seuils adaptatifs
    MIN_WIDTH_PX = 1200          # En dessous → on upscale
    SHARPNESS_THRESHOLD = 75     # Score en dessous duquel on applique l'unsharp mask
    CONTRAST_THRESHOLD = 75      # Score en dessous duquel on applique CLAHE
    MAX_DESKEW_ANGLE = 15.0      # Angle max corrigé (au-delà = probablement fausse détection)
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID = (8, 8)

    def __init__(self, save_debug: bool = True, debug_dir: str = "outputs/preprocessed"):
        self.save_debug = save_debug
        self.debug_dir = debug_dir
        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    # ==================================================
    # POINT D'ENTRÉE PRINCIPAL
    # ==================================================

    def preprocess(self, image_path: str, quality_metrics: dict) -> np.ndarray:
        """
        Charge et prétraite l'image selon les métriques de qualité fournies.

        Args:
            image_path:       Chemin vers l'image source.
            quality_metrics:  Dictionnaire retourné par ImageQualityChecker.check().

        Returns:
            Image BGR (np.ndarray) prête pour PaddleOCR.
        """
        image = self._load_image(image_path)

        sharpness_score = quality_metrics.get("text_sharpness", {}).get("score", 100)
        contrast_score  = quality_metrics.get("local_text_contrast", {}).get("score", 100)

        print(f"  [Preprocess] Résolution initiale : {image.shape[1]}x{image.shape[0]}")

        # --- Étape 1 : Upscale si nécessaire ---
        image = self._upscale_if_needed(image)

        # --- Étape 2 : Débruitage (léger, toujours) ---
        image = self._denoise(image)

        # --- Étape 3 : CLAHE (contraste adaptatif) ---
        if contrast_score < self.CONTRAST_THRESHOLD:
            print(f"  [Preprocess] Contraste faible ({contrast_score}) → application CLAHE")
            image = self._apply_clahe(image)

        # --- Étape 4 : Unsharp mask (netteté) ---
        if sharpness_score < self.SHARPNESS_THRESHOLD:
            print(f"  [Preprocess] Netteté insuffisante ({sharpness_score}) → unsharp mask")
            image = self._apply_unsharp_mask(image)

        # --- Étape 5 : Deskew (correction rotation) ---
        image = self._deskew(image)

        print(f"  [Preprocess] Résolution finale : {image.shape[1]}x{image.shape[0]}")

        # --- Étape 6 : Sauvegarde debug ---
        if self.save_debug:
            self._save_debug(image, image_path)

        return image

    # ==================================================
    # CHARGEMENT
    # ==================================================

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Charge l'image en respectant l'orientation EXIF."""
        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        rgb_array = np.array(pil_image.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # ==================================================
    # ÉTAPE 1 : UPSCALE
    # ==================================================

    def _upscale_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < self.MIN_WIDTH_PX:
            scale = self.MIN_WIDTH_PX / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w != w or new_h != h:
                print(f"  [Preprocess] Upscale {w}x{h} -> {new_w}x{new_h}")
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    # ==================================================
    # ÉTAPE 2 : DÉBRUITAGE
    # ==================================================

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """
        Débruitage non-local means (léger).
        h=5 est volontairement conservateur pour ne pas lisser les détails fins du texte.
        """
        return cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                               templateWindowSize=7, searchWindowSize=21)

    # ==================================================
    # ÉTAPE 3 : CLAHE (contraste adaptatif)
    # ==================================================

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applique CLAHE sur le canal L de l'espace LAB pour améliorer
        le contraste sans saturer les couleurs.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_TILE_GRID
        )
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ==================================================
    # ÉTAPE 4 : UNSHARP MASK (netteté)
    # ==================================================

    @staticmethod
    def _apply_unsharp_mask(image: np.ndarray,
                             strength: float = 1.5,
                             blur_sigma: float = 1.0) -> np.ndarray:
        """
        Accentue les contours des caractères via unsharp masking.
        strength : intensité de l'accentuation (1.0 = neutre, 2.0 = fort)
        """
        blurred = cv2.GaussianBlur(image, (0, 0), blur_sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    # ==================================================
    # ÉTAPE 5 : DESKEW
    # ==================================================

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte et corrige la rotation de l'image via les lignes de Hough
        sur les bords du texte (Canny). Ne corrige que les angles faibles
        (< MAX_DESKEW_ANGLE) pour éviter les fausses détections.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=image.shape[1] // 6,
            maxLineGap=20
        )

        if lines is None:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Filtrer les lignes quasi-horizontales (texte) uniquement
                if abs(angle) < self.MAX_DESKEW_ANGLE:
                    angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.3:
            # Correction négligeable
            return image

        print(f"  [Preprocess] Deskew : angle détecté = {median_angle:.2f}°")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    # ==================================================
    # ÉTAPE 6 : SAUVEGARDE DEBUG
    # ==================================================

    def _save_debug(self, image: np.ndarray, original_path: str) -> None:
        """
        Sauvegarde l'image prétraitée dans outputs/preprocessed/
        en conservant le nom du fichier source.
        """
        stem = Path(original_path).stem
        suffix = Path(original_path).suffix or ".png"
        debug_filename = f"{stem}_preprocessed{suffix}"
        debug_path = os.path.join(self.debug_dir, debug_filename)

        # Convertir BGR → RGB pour PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.save(debug_path)

        print(f"  [Preprocess] Image prétraitée sauvegardée → {debug_path}")
