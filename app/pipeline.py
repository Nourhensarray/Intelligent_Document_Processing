from app.ocr.paddle_engine import PaddleEngine
from app.extraction.value_extractor import ValueExtractor
from app.extraction.field_matcher import FieldMatcher
from app.quality.image_quality_checker import ImageQualityChecker
from app.preprocessing.image_preprocessor import ImagePreprocessor
from app.postprocessing.ocr_postprocessor import OCRPostProcessor
from app.layout.document_layout import DocumentLayoutBuilder


class DocumentPipeline:
    """
    Pipeline complet de traitement d'un document.

    Étapes :
        0. Vérification qualité
        0.5 Prétraitement
        1. OCR PaddleOCR
        1.5 Post-traitement OCR
        1.6 Construction du layout
        2. Extraction
    """

    def __init__(self, fast_mode: bool = False):

        print("Initialisation du moteur OCR...")

        self.quality_checker = ImageQualityChecker()

        self.preprocessor = ImagePreprocessor(
            save_debug=False,
            debug_dir="outputs/preprocessed",
            fast_mode=fast_mode
        )

        self.ocr_engine = PaddleEngine()

        self.postprocessor = OCRPostProcessor()

        self.layout_builder = DocumentLayoutBuilder()

        # Extracteur unifié (MRZ + OCR Spatial) — partage l'instance FieldMatcher
        # pour éviter l'initialisation double et profiter du cache commun.
        self.field_matcher = FieldMatcher()
        self.value_extractor = ValueExtractor(matcher=self.field_matcher)

        # Champs principaux attendus sur tout document d'identité.
        # Au moins MIN_MAIN_FIELDS d'entre eux doivent être extraits avec
        # une valeur non-vide pour qu'un document soit considéré valide.
        self._main_field_names = {
            "nom",
            "prenom",
            "numero_document",
            "nationalite",
            "date_naissance",
        }
        self._MIN_MAIN_FIELDS = 2   # seuil minimum de champs principaux valides

    # ------------------------------------------------------------------
    # VALIDATION POST-EXTRACTION
    # ------------------------------------------------------------------

    def _has_clear_main_key_values(self, layout: list, extracted_data: dict) -> bool:
        """
        Pour chaque label principal (nom, prenom, numero_document, nationalite,
        date_naissance) détecté dans le layout OCR, vérifie que sa valeur
        correspondante est correctement extraite dans extracted_data.

        Règles :
        - Si un label principal est présent dans le document mais sa valeur
          n'est pas extraite (vide ou < 2 chars) → FAILED immédiat.
        - Si aucun label principal n'est détecté dans le layout (passeport MRZ pur
          ou format non structuré), on tombe sur le seuil minimum de champs extraits.
        - Le FieldMatcher utilise son cache interne → pas de fuzzy matching redondant.
        """
        if not extracted_data:
            return False

        # ── Étape 1 : collecter les labels principaux présents dans le layout ──
        detected_labels: set = set()
        for line in layout:
            for item in (line.get("items") or []):
                text = item.get("text", "")
                if not text:
                    continue
                field, _ = self.field_matcher.match_with_keyword(text)
                if field and field in self._main_field_names:
                    detected_labels.add(field)

        # ── Étape 2 : aucun label détecté (MRZ pur / format inconnu) ──
        # → seuil minimum sur les champs extraits pour valider quand même
        if not detected_labels:
            valid_count = sum(
                1 for f in self._main_field_names
                if extracted_data.get(f) and len(str(extracted_data.get(f, "")).strip()) >= 2
            )
            return valid_count >= self._MIN_MAIN_FIELDS

        # ── Étape 3 : pour chaque label détecté, la valeur doit être extraite ──
        for field in detected_labels:
            value = extracted_data.get(field, "")
            if not value or len(str(value).strip()) < 2:
                # Label présent dans le document, valeur absente → document invalide
                return False

        return True


    def process(self, image_path):

        # =========================================
        # ÉTAPE 0 : QUALITÉ
        # =========================================

        print("Étape 0 : Vérification qualité d'image...")

        quality = self.quality_checker.check(image_path)

        print(f"Score qualité : {quality['score']}/100")
        print(f"Statut qualité : {quality['status']}")

        if quality["status"] not in ["ACCEPTED", "WARNING"]:

            print(
                f"[X] Image non acceptée ({quality['status']}) - OCR annulé"
            )

            if quality["reasons"]:
                print("Raisons :")
                for reason in quality["reasons"]:
                    print(f"- {reason}")

            return {
                "status": quality["status"],
                "quality": quality,
                "preprocessing": None,
                "ocr": None,
                "layout": None,
                "data_old": None,
                "data_new": None,
            }

        print("[OK] Image acceptée")

        # =========================================
        # ÉTAPE 0.5 : PRÉTRAITEMENT
        # =========================================

        print("Étape 0.5 : Prétraitement...")

        preprocessed_image = self.preprocessor.preprocess(
            image_path=image_path,
            quality_metrics=quality["metrics"]
        )

        print("[OK] Prétraitement terminé")

        # =========================================
        # ÉTAPE 1 : OCR
        # =========================================

        print("Étape 1 : OCR...")

        ocr_data = self.ocr_engine.extract(preprocessed_image)

        print(f"{len(ocr_data)} zones de texte détectées")

        # =========================================
        # ÉTAPE 1.5 : POST-TRAITEMENT OCR
        # =========================================

        print("Étape 1.5 : Nettoyage OCR...")

        ocr_data = self.postprocessor.process(ocr_data)

        # =========================================
        # ÉTAPE 1.6 : CONSTRUCTION DU LAYOUT
        # =========================================

        print("Étape 1.6 : Construction du layout...")

        layout = self.layout_builder.build(ocr_data)

        print(f"{len(layout)} lignes construites")

        # =========================================
        # ÉTAPE 2 : EXTRACTION
        # =========================================

        print("Étape 2 : Extraction des valeurs...")

        extracted_data = self.value_extractor.extract(layout)

        if not self._has_clear_main_key_values(layout, extracted_data):
            print("[X] Image rejetée : label(s) principal/aux détecté(s) dans l'OCR sans valeur extraite")
            return {
                "status": "FAILED",
                "failure_code": "REJECTED_UNCLEAR_KEYS",
                "quality": quality,
                "preprocessing": {
                    "applied": True,
                    "debug_saved": self.preprocessor.save_debug,
                    "debug_dir": self.preprocessor.debug_dir,
                },
                "ocr": ocr_data,
                "layout": layout,
                "data": extracted_data,
            }

        # Validation of completeness (to reject "une seule partie" / partial documents)
        # If we couldn't extract at least 3 fields, consider the document incomplete.
        if len(extracted_data) < 3:
            print("[X] Image rejetée : Document incomplet (moins de 3 champs extraits)")
            return {
                "status": "FAILED",
                "failure_code": "REJECTED_INCOMPLETE",
                "quality": quality,
                "preprocessing": {
                    "applied": True,
                    "debug_saved": self.preprocessor.save_debug,
                    "debug_dir": self.preprocessor.debug_dir,
                },
                "ocr": ocr_data,
                "layout": layout,
                "data": extracted_data,
            }

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

            "layout": layout,

            # Résultat final de l'extraction
            "data": extracted_data,
        }
