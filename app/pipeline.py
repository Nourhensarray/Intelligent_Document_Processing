from app.ocr.paddle_engine import PaddleEngine
from app.extraction.value_extractor import ValueExtractor
from app.quality.image_quality_checker import ImageQualityChecker
from app.preprocessing.image_preprocessor import ImagePreprocessor


class DocumentPipeline:
    """
    Pipeline complet de traitement d'un document :

    1. Vérification de la qualité de l'image (focus lisibilité texte)
    2. Prétraitement adaptatif (deskew, CLAHE, débruitage, netteté, upscale)
    3. OCR avec PaddleOCR sur l'image prétraitée
    4. Extraction des données clé-valeur
    """

    def __init__(self):

        print("Initialisation du moteur OCR...")

        self.quality_checker = ImageQualityChecker()
        self.preprocessor    = ImagePreprocessor(save_debug=True, debug_dir="outputs/preprocessed")
        self.ocr_engine      = PaddleEngine()
        self.value_extractor = ValueExtractor()

    def process(self, image_path):

        # =========================================
        # ÉTAPE 0 : CONTRÔLE DE LA QUALITÉ
        # =========================================

        print("Étape 0 : Vérification qualité d'image...")

        quality = self.quality_checker.check(image_path)

        print(f"Score qualité : {quality['score']}/100")
        print(f"Statut qualité : {quality['status']}")

        # =========================================
        # IMAGE NON ACCEPTÉE → arrêt immédiat
        # =========================================

        if quality["status"] != "ACCEPTED":

            print("[X] Image non acceptée (status: {0}) - OCR et extraction annulés".format(quality["status"]))

            if quality["reasons"]:
                print("Raisons :")
                for reason in quality["reasons"]:
                    print(f"- {reason}")

            return {
                "status": quality["status"],
                "quality": quality,
                "preprocessing": None,
                "ocr": None,
                "data": None,
            }

        print("[OK] Image acceptée")

        # =========================================
        # ÉTAPE 0.5 : PRÉTRAITEMENT
        # =========================================

        print("Étape 0.5 : Prétraitement de l'image...")

        preprocessed_image = self.preprocessor.preprocess(
            image_path=image_path,
            quality_metrics=quality["metrics"]
        )

        print("[OK] Prétraitement terminé")

        # =========================================
        # ÉTAPE 1 : OCR (sur image prétraitée)
        # =========================================

        print("Étape 1 : OCR...")

        ocr_data = self.ocr_engine.extract(preprocessed_image)

        print(f"{len(ocr_data)} zones de texte détectées")

        # =========================================
        # ÉTAPE 2 : EXTRACTION CLÉ-VALEUR
        # =========================================

        print("Étape 2 : Extraction clé-valeur...")

        data = self.value_extractor.extract(ocr_data)

        # =========================================
        # RÉSULTAT FINAL
        # =========================================

        return {
            "status": "SUCCESS",
            "quality": quality,
            "preprocessing": {
                "applied": True,
                "debug_saved": self.preprocessor.save_debug,
                "debug_dir": self.preprocessor.debug_dir,
            },
            "ocr": ocr_data,
            "data": data,
        }