import cv2
import numpy as np
from PIL import Image


class ImageQualityChecker:
    """
    Vérifie la qualité d'une image de document pour l'OCR en se basant
    principalement sur la qualité et la lisibilité du texte.

    Critères évalués :
    1. Netteté des contours de caractères (Text Edge Sharpness - 30%)
    2. Contraste local Texte / Arrière-plan (Local Text Contrast - 25%)
    3. Lisibilité & Qualité de binarisation Otsu (Binarization Quality - 20%)
    4. Résolution & taille des caractères (Resolution - 15%)
    5. Présence et densité de structures textuelles (Text Density - 5%)
    6. Reflets et éblouissements (Reflections - 3%)
    7. Luminosité générale (Brightness - 2%)

    Garde-fous automatiques (REJECTED immédiat) :
    - Résolution insuffisante : largeur < MIN_WIDTH ou hauteur < MIN_HEIGHT
    - Hauteur moyenne des caractères < MIN_CHAR_HEIGHT_PX (texte illisible par l'OCR)
    - Netteté, contraste ou binarisation sous le critical_floor
    """

    # Résolution minimale ABSOLUE (images vraiment inutilisables)
    MIN_WIDTH  = 200    # px — en dessous, même l'œil humain ne peut pas lire
    MIN_HEIGHT = 150    # px
    # Hauteur minimale des caractères (garde-fou extrême)
    MIN_CHAR_HEIGHT_PX = 4

    def __init__(
        self,
        accept_threshold: int = 70,
        warning_threshold: int = 55,
        critical_floor: int = 45,
    ):
        self.accept_threshold = accept_threshold
        self.warning_threshold = warning_threshold
        self.critical_floor = critical_floor

    # ==================================================
    # POINT D'ENTRÉE PRINCIPAL
    # ==================================================

    def check(self, image_path: str) -> dict:
        image = self._load_image(image_path)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Détection des régions de texte ---
        stroke_mask, bg_mask, text_density = self._extract_text_masks(gray)

        # --- 1. Netteté des contours de caractères ---
        text_sharpness_val, text_sharpness_score = self._calculate_text_edge_sharpness(gray, stroke_mask)

        # --- 2. Contraste local Texte / Arrière-plan (Binarisation Otsu) ---
        local_contrast_val, local_contrast_score = self._calculate_local_text_contrast(gray)

        # --- 3. Qualité de binarisation & Séparabilité ---
        binarization_val, binarization_score = self._calculate_binarization_quality(gray)

        # --- 4. Résolution & taille réelle des caractères ---
        char_height_px = self._estimate_char_height(gray)
        resolution_score = self._calculate_resolution_score(width, height, char_height_px)

        # --- 5. Densité de structures textuelles ---
        text_density_score = self._calculate_text_density_score(text_density)

        # --- 6. Reflets ---
        reflection_ratio = self._calculate_reflection_ratio(image)
        reflection_score = self._calculate_reflection_score(reflection_ratio)

        # --- 7. Luminosité ---
        brightness_val = float(gray.mean())
        brightness_score = self._calculate_brightness_score(brightness_val)

        # --- Score global pondéré ---
        weighted_score = (
            text_sharpness_score  * 0.30
            + local_contrast_score  * 0.25
            + binarization_score    * 0.20
            + resolution_score      * 0.15
            + text_density_score    * 0.05
            + reflection_score      * 0.03
            + brightness_score      * 0.02
        )

        # --- Garde-fou anti-compensation sur les métriques clés ---
        # Si une métrique critique est très faible, on plafonne le score global
        # pour éviter qu'un excellent contraste compense une résolution inutilisable.
        # EXCEPTION : Si le texte est mathématiquement parfait (image de synthèse ou scan parfait),
        # on ne plafonne pas sur la résolution.
        is_perfect_text = (binarization_val > 0.85 and text_sharpness_val > 4000 and local_contrast_val > 150)
        
        if is_perfect_text:
            critical_text_min = min(text_sharpness_score, local_contrast_score, binarization_score)
        else:
            critical_text_min = min(text_sharpness_score, local_contrast_score, binarization_score, resolution_score)
            
        score = min(weighted_score, critical_text_min + 15)
        score = int(round(max(0, min(100, score))))

        # --- Détermination des raisons ---
        reasons = []
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            reasons.append(
                f"Résolution trop faible ({width}x{height} px) — minimum requis : "
                f"{self.MIN_WIDTH}x{self.MIN_HEIGHT} px"
            )
        if char_height_px < self.MIN_CHAR_HEIGHT_PX:
            reasons.append(
                f"Taille des caractères trop petite ({char_height_px:.1f} px) — "
                f"minimum requis : {self.MIN_CHAR_HEIGHT_PX} px"
            )
        if text_sharpness_score < 60:
            reasons.append("Netteté des caractères insuffisante (texte flou)")
        if local_contrast_score < 60:
            reasons.append("Contraste local entre le texte et l'arrière-plan trop faible")
        if binarization_score < 60:
            reasons.append("Lisibilité de la binarisation du texte dégradée")
        if resolution_score < 60:
            reasons.append("Résolution ou taille des caractères insuffisante pour une extraction fiable")
        if text_density_score < 60:
            reasons.append("Densité de structures textuelles très faible (texte absent ou illisible)")
        if reflection_score < 60:
            reasons.append("Reflets ou éblouissements importants détectés sur le document")
        if brightness_score < 60:
            reasons.append("Luminosité globale inadaptée (" + ("trop sombre" if brightness_val < 60 else "trop claire") + ")")

        # --- Garde-fous absolus → REJECTED immédiat ---
        # On ne rejette que ce qui est VRAIMENT illisible :
        # taille microscopique, texte intrinsèquement flou/absent.
        # La résolution insuffisante contribue au score mais ne rejette
        # pas seule : une petite image nette peut être utilisable.
        hard_reject = (
            width < self.MIN_WIDTH
            or height < self.MIN_HEIGHT
            or char_height_px < self.MIN_CHAR_HEIGHT_PX
            or critical_text_min < self.critical_floor
            or text_density < 0.001
        )

        if hard_reject:
            status = "REJECTED"
        elif score >= self.accept_threshold:
            status = "ACCEPTED"
        elif score >= self.warning_threshold:
            status = "WARNING"
        else:
            status = "REJECTED"

        return {
            "status": status,
            "score": score,
            "metrics": {
                "text_sharpness": {
                    "laplacian_variance": round(text_sharpness_val, 2),
                    "score": text_sharpness_score,
                },
                "local_text_contrast": {
                    "difference": round(local_contrast_val, 2),
                    "score": local_contrast_score,
                },
                "binarization_quality": {
                    "otsu_separation": round(binarization_val, 4),
                    "score": binarization_score,
                },
                "resolution": {
                    "width": width,
                    "height": height,
                    "char_height_px": round(char_height_px, 1),
                    "score": resolution_score,
                },
                "text_density": {
                    "density_ratio": round(text_density, 4),
                    "score": text_density_score,
                },
                "reflections": {
                    "ratio": round(reflection_ratio, 4),
                    "score": reflection_score,
                },
                "brightness": {
                    "value": round(brightness_val, 2),
                    "score": brightness_score,
                },
            },
            "reasons": reasons,
        }

    # ==================================================
    # MASQUAGE ET DÉTECTION DU TEXTE
    # ==================================================

    def _extract_text_masks(self, gray: np.ndarray):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.Canny(gray, 60, 160)

        stroke_mask = cv2.dilate(edges, kernel, iterations=1) > 0

        bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        expanded_mask = cv2.dilate(stroke_mask.astype(np.uint8), bg_kernel, iterations=1) > 0
        bg_mask = expanded_mask & (~stroke_mask)

        text_density = float(np.sum(stroke_mask)) / stroke_mask.size
        return stroke_mask, bg_mask, text_density

    # ==================================================
    # NETTETÉ LOCALE DU TEXTE
    # ==================================================

    def _calculate_text_edge_sharpness(self, gray: np.ndarray, stroke_mask: np.ndarray):
        if stroke_mask.sum() < 20:
            return 0.0, 20

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_values = laplacian[stroke_mask]
        sharpness_val = float(local_values.var())

        if sharpness_val >= 400:
            score = 100
        elif sharpness_val >= 200:
            score = 90
        elif sharpness_val >= 100:
            score = 75
        elif sharpness_val >= 50:
            score = 55
        elif sharpness_val >= 25:
            score = 40
        else:
            score = 20

        return sharpness_val, score

    # ==================================================
    # CONTRASTE LOCAL TEXTE vs FOND (Seuillage Otsu)
    # ==================================================

    def _calculate_local_text_contrast(self, gray: np.ndarray):
        """
        Mesure la différence d'intensité moyenne entre les pixels du texte (sombre)
        et les pixels du fond (clair) séparés par la binarisation d'Otsu.
        """
        thresh_val, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        fg_pixels = gray[binarized == 0]
        bg_pixels = gray[binarized == 255]

        if fg_pixels.size == 0 or bg_pixels.size == 0:
            return 0.0, 20

        contrast_diff = abs(float(bg_pixels.mean()) - float(fg_pixels.mean()))

        if contrast_diff >= 55:
            score = 100
        elif contrast_diff >= 40:
            score = 90
        elif contrast_diff >= 25:
            score = 75
        elif contrast_diff >= 15:
            score = 55
        elif contrast_diff >= 8:
            score = 40
        else:
            score = 20

        return contrast_diff, score

    # ==================================================
    # QUALITÉ DE BINARISATION OTSU
    # ==================================================

    def _calculate_binarization_quality(self, gray: np.ndarray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total_pixels = gray.size
        if total_pixels == 0:
            return 0.0, 20

        norm_hist = hist / total_pixels
        cumsum = np.cumsum(norm_hist)
        cummean = np.cumsum(norm_hist * np.arange(256))

        global_mean = cummean[-1]
        global_var = np.sum(norm_hist * ((np.arange(256) - global_mean) ** 2))

        if global_var == 0:
            return 0.0, 20

        w0 = cumsum
        w1 = 1.0 - w0
        valid = (w0 > 0) & (w1 > 0)

        mean0 = np.zeros(256)
        mean1 = np.zeros(256)
        mean0[valid] = cummean[valid] / w0[valid]
        mean1[valid] = (global_mean - cummean[valid]) / w1[valid]

        between_var = w0 * w1 * ((mean0 - mean1) ** 2)
        max_between_var = float(np.max(between_var))

        eta = float(max_between_var / global_var)

        if eta >= 0.50:
            score = 100
        elif eta >= 0.35:
            score = 85
        elif eta >= 0.25:
            score = 70
        elif eta >= 0.15:
            score = 55
        elif eta >= 0.08:
            score = 40
        else:
            score = 20

        return eta, score

    # ==================================================
    # DENSITÉ DE STRUCTURES TEXTUELLES
    # ==================================================

    def _calculate_text_density_score(self, text_density: float) -> int:
        if text_density >= 0.015:
            return 100
        elif text_density >= 0.008:
            return 85
        elif text_density >= 0.004:
            return 70
        elif text_density >= 0.001:
            return 50
        else:
            return 20

    # ==================================================
    # REFLETS ET ÉBLOUISSEMENTS
    # ==================================================

    def _calculate_reflection_ratio(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]

        std_val = float(value.std())
        mean_val = float(value.mean())

        if std_val < 25.0 and mean_val > 200:
            return 0.0

        reflection_mask = (value > 250) & (saturation < 20) & (value > (mean_val + 35))
        return float(np.sum(reflection_mask)) / reflection_mask.size

    def _calculate_reflection_score(self, ratio: float) -> int:
        if ratio < 0.05:
            return 100
        elif ratio < 0.15:
            return 85
        elif ratio < 0.30:
            return 70
        elif ratio < 0.50:
            return 45
        else:
            return 20

    # ==================================================
    # LUMINOSITÉ GÉNÉRALE
    # ==================================================

    def _calculate_brightness_score(self, brightness_val: float) -> int:
        if 70 <= brightness_val <= 250:
            return 100
        elif 50 <= brightness_val < 70 or 250 < brightness_val <= 253:
            return 85
        elif 35 <= brightness_val < 50 or 253 < brightness_val <= 255:
            return 70
        else:
            return 30

    # ==================================================
    # RÉSOLUTION ET TAILLE DES CARACTÈRES
    # ==================================================

    def _calculate_resolution_score(self, width: int, height: int, char_height_px: float) -> int:
        """
        Note la résolution et la taille réelle des caractères.

        Deux sous-critères :
        - Résolution globale (largeur x hauteur)
        - Hauteur moyenne des caractères détectés (en pixels)

        Le score est une combinaison pondérée des deux :
        - 60% résolution globale (pixels totaux)
        - 40% hauteur des caractères détectés
        Une petite image avec du texte très net obtient un score correct (~60-70).
        """
        # Score résolution globale
        pixels = width * height
        if pixels >= 2_000_000:        # ≥ 2 Mpx
            res_score = 100
        elif pixels >= 1_000_000:      # ≥ 1 Mpx
            res_score = 90
        elif pixels >= 500_000:        # ≥ 500 Kpx (ex: 1000×500)
            res_score = 80
        elif pixels >= 250_000:        # ≥ 250 Kpx (ex: 800×310)
            res_score = 70
        elif pixels >= 120_000:        # ≥ 120 Kpx (ex: 619×350 ← cin3)
            res_score = 60
        elif pixels >= 60_000:         # ≥ 60 Kpx  (ex: 400×294 ← cin2)
            res_score = 45
        elif pixels >= 30_000:         # ≥ 30 Kpx
            res_score = 30
        else:
            res_score = 10

        # Score hauteur des caractères (proxy de la lisibilité réelle)
        if char_height_px >= 20:
            char_score = 100
        elif char_height_px >= 14:
            char_score = 90
        elif char_height_px >= 10:
            char_score = 80
        elif char_height_px >= 7:
            char_score = 65
        elif char_height_px >= self.MIN_CHAR_HEIGHT_PX:
            char_score = 45
        else:
            char_score = 10

        # Score final = combinaison pondérée (résolution + taille caractères)
        return int(round(res_score * 0.6 + char_score * 0.4))

    def _estimate_char_height(self, gray: np.ndarray) -> float:
        """
        Estime la hauteur médiane des caractères en pixels.

        Stratégie robuste :
        - On travaille sur le crop central (84%) de l'image pour éviter le fond/table
        - Binarisation Otsu inversée → composantes connexes
        - Filtrage strict : ratio forme caractère, taille raisonnable, remplissage > 15%
        - On retient les 200 composantes avec la plus grande aire (les vraies lettres)
          et on prend leur hauteur médiane.
        """
        img_h, img_w = gray.shape
        # Crop central 84% (élimine les bords et le fond)
        margin_x = int(img_w * 0.08)
        margin_y = int(img_h * 0.08)
        crop = gray[margin_y:img_h - margin_y, margin_x:img_w - margin_x]

        _, binarized = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binarized, connectivity=8)

        if num_labels <= 1:
            return 0.0

        img_area = crop.shape[0] * crop.shape[1]
        candidates = []

        for i in range(1, num_labels):
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            # Exclure composantes trop grandes (logo, photo, fond)
            if area > img_area * 0.02:
                continue
            # Exclure bruit pixel unique
            if h < 3 or w < 2:
                continue
            # Forme de caractère : ratio largeur/hauteur entre 0.15 et 6
            ratio = w / h
            if ratio < 0.15 or ratio > 6:
                continue
            # Remplissage > 15% (lettres solides, pas des contours vides)
            fill = area / (w * h) if (w * h) > 0 else 0
            if fill < 0.15:
                continue

            candidates.append((area, float(h)))

        if not candidates:
            return 0.0

        # Trier par aire décroissante et prendre les 200 plus grandes composantes
        candidates.sort(key=lambda x: -x[0])
        top = candidates[:200]
        heights = [h for _, h in top]

        return float(np.median(heights))

    # ==================================================
    # CHARGEMENT D'IMAGE
    # ==================================================

    def _load_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            array = np.array(rgb_image)
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)