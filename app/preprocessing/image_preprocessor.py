import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


class ImagePreprocessor:
    """
    Prétraitement adaptatif d'une image de document après validation qualité.

    Étapes appliquées conditionnellement selon les métriques de qualité :
     0. Crop document     – détection et recadrage automatique du document
     1. Redimensionnement  – si la résolution est insuffisante pour l'OCR
     2. Débruitage léger   – Gaussian blur rapide (remplace NLMeans très lent)
     3. CLAHE              – si le contraste local est faible
     4. Unsharp mask       – si la netteté des caractères est insuffisante
     5. Deskew             – correction de la rotation (Hough lines)
     6. Sauvegarde debug   – dans outputs/preprocessed/ (optionnel)

    Performance :
    - L'ancien fastNlMeansDenoisingColored (searchWindowSize=21) prenait ~10-15s/image.
    - Le nouveau pipeline prend < 0.5s/image sur les mêmes images.
    - Activez fast_mode=True pour bypasser tout le prétraitement (image brute → OCR direct).
    """

    # Seuils adaptatifs
    MIN_WIDTH_PX = 1200          # En dessous → on upscale
    MAX_WIDTH_PX = 1600          # Au dessus → on downscale pour accélérer l'OCR
    SHARPNESS_THRESHOLD = 75     # Score en dessous duquel on applique l'unsharp mask
    CONTRAST_THRESHOLD = 75      # Score en dessous duquel on applique CLAHE
    MAX_DESKEW_ANGLE = 15.0      # Angle max corrigé (au-delà = probablement fausse détection)
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID = (8, 8)

    def __init__(self,
                 save_debug: bool = False,
                 debug_dir: str = "outputs/preprocessed",
                 fast_mode: bool = False):
        """
        Args:
            save_debug : Sauvegarder l'image prétraitée pour inspection (désactivé par défaut
                         car le I/O disque ralentit le traitement par lot).
            debug_dir  : Répertoire de sauvegarde des images debug.
            fast_mode  : Si True, bypasse TOUT le prétraitement et retourne l'image brute.
                         Utile pour mesurer la vitesse OCR seule, ou pour des images
                         de très bonne qualité qui n'en ont pas besoin.
        """
        self.save_debug = save_debug
        self.debug_dir = debug_dir
        self.fast_mode = fast_mode
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

        # --- Mode rapide : aucun prétraitement ---
        if self.fast_mode:
            print("  [Preprocess] Mode rapide active -> aucun pretraitement applique")
            return image

        sharpness_score = quality_metrics.get("text_sharpness", {}).get("score", 100)
        contrast_score  = quality_metrics.get("local_text_contrast", {}).get("score", 100)

        print(f"  [Preprocess] Résolution initiale : {image.shape[1]}x{image.shape[0]}")

        # --- Étape 0 : Détection et recadrage du document ---
        image = self._crop_document(image)

        # --- Étape 1 : Redimensionnement si nécessaire ---
        image = self._resize_if_needed(image)

        # --- Étape 2 : Débruitage rapide (Gaussian blur léger) ---
        # NOTE : L'ancien fastNlMeansDenoisingColored (searchWindowSize=21) était
        # extrêmement lent (~10-15s/image). On le remplace par un Gaussian blur
        # avec kernel 3x3 qui est quasi-instantané et suffisant pour l'OCR.
        image = self._denoise_fast(image)

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
    # ÉTAPE 0 : DÉTECTION ET RECADRAGE DU DOCUMENT
    # ==================================================

    def _crop_document(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte le plus grand rectangle (document) dans l'image et le recadre.
        Utilise un système de scoring basé sur :
        - Le ratio d'aspect (proche d'un document standard)
        - La surface relative dans l'image
        - La rectangularité du contour (remplissage du bounding rect)

        Si aucun document n'est trouvé, retourne l'image originale.
        """
        h, w = image.shape[:2]
        img_area = h * w

        # Convertir en niveaux de gris et appliquer un flou pour réduire le bruit
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection de contours via Canny + dilatation pour fermer les bords
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Trier par aire décroissante
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Ratios d'aspect typiques de documents d'identité
        # Passeport ouvert (2 pages) : ~1.4:1 en paysage
        # Passeport fermé : ~0.7:1 en portrait (ou 1.42:1 en paysage)
        # Carte d'identité : ~1.58:1 (format CR-80)
        TYPICAL_RATIOS = [1.42, 0.70, 1.58, 1.0, 0.63]

        best_score = -1
        best_rect = None

        for cnt in contours[:15]:
            area = cv2.contourArea(cnt)
            ratio = area / img_area

            # Le document doit représenter entre 3% et 85% de l'image
            if ratio < 0.03 or ratio > 0.85:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            # Ignorer les contours trop fins (bandes, lignes)
            if cw < w * 0.1 or ch < h * 0.1:
                continue

            # Calculer le ratio d'aspect du bounding rect
            aspect = max(cw, ch) / max(min(cw, ch), 1)

            # Score de similarité avec les ratios de documents standards
            aspect_scores = [1.0 / (1.0 + abs(aspect - r)) for r in TYPICAL_RATIOS]
            aspect_score = max(aspect_scores)

            # Rectangularité : surface du contour / surface du bounding rect
            rect_area = cw * ch
            rectangularity = area / max(rect_area, 1)

            # Score final : aspect_score * rectangularity * ratio_bonus
            # ratio_bonus favorise les contours ni trop petits ni trop grands
            ratio_bonus = min(ratio / 0.10, 1.0)  # plateau à 10%

            score = aspect_score * rectangularity * ratio_bonus

            if score > best_score:
                best_score = score
                best_rect = (x, y, cw, ch)

        if best_rect is None or best_score < 0.40:
            return image

        x, y, cw, ch = best_rect
        # Ajouter un petit padding (2%)
        pad_x = int(cw * 0.02)
        pad_y = int(ch * 0.02)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)

        cropped = image[y1:y2, x1:x2]
        print(f"  [Preprocess] Document détecté (score={best_score:.2f}) : crop {x2-x1}x{y2-y1}")
        return cropped

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
    # ÉTAPE 1 : REDIMENSIONNEMENT
    # ==================================================

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < self.MIN_WIDTH_PX:
            scale = self.MIN_WIDTH_PX / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"  [Preprocess] Upscale {w}x{h} -> {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif w > self.MAX_WIDTH_PX:
            scale = self.MAX_WIDTH_PX / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"  [Preprocess] Downscale {w}x{h} -> {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    # ==================================================
    # ÉTAPE 2 : DÉBRUITAGE RAPIDE
    # ==================================================

    @staticmethod
    def _denoise_fast(image: np.ndarray) -> np.ndarray:
        """
        Débruitage rapide par Gaussian blur léger (kernel 3x3, sigma=0.8).

        Pourquoi ce changement ?
        - fastNlMeansDenoisingColored avec searchWindowSize=21 prenait 10-15s/image
          soit ~12 min pour 50 images (goulot d'étranglement principal).
        - Le Gaussian blur 3x3 prend < 5ms/image, soit x2000 plus rapide.
        - Pour l'OCR de passeports (texte imprimé net), un léger blur suffit
          amplement pour atténuer le bruit de capteur/compression JPEG.
        """
        return cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)

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
        
        # Calculer les nouvelles dimensions pour éviter le recadrage (crop)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Ajuster la matrice de rotation pour le décalage de centre
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
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

        # Convertir BGR -> RGB pour PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.save(debug_path)

        print(f"  [Preprocess] Image prétraitée sauvegardée -> {debug_path}")
