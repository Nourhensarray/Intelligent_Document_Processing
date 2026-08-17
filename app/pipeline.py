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

        # ── Champs OBLIGATOIRES ──
        # Ces 4 champs doivent TOUS être présents avec une valeur non-vide
        # pour qu'un document soit considéré valide (SUCCESS).
        # Si l'un d'eux manque → FAILED, aucune donnée écrite dans la table.
        self._mandatory_fields = {
            "nom",
            "prenom",
            "numero_document",
            "nationalite",
        }

        # Champs principaux supplémentaires (utilisés pour la validation
        # secondaire basée sur le layout OCR).
        self._main_field_names = self._mandatory_fields | {"date_naissance"}

    # ------------------------------------------------------------------
    # VALIDATION POST-EXTRACTION
    # ------------------------------------------------------------------

    def _has_clear_main_key_values(self, layout: list, extracted_data: dict) -> bool:
        """
        Valide que les données extraites contiennent les champs obligatoires.

        Règle stricte :
        - Les 4 champs obligatoires (nom, prenom, numero_document, nationalite)
          doivent TOUS avoir une valeur non-vide (≥ 2 caractères).
        - Si l'un d'eux est absent ou vide → FAILED immédiat.

        Règle secondaire (layout) :
        - Pour chaque label principal détecté dans le layout OCR, sa valeur
          correspondante doit être correctement extraite.
        """
        if not extracted_data:
            return False

        # ── Étape 1 (STRICTE) : vérifier les 4 champs obligatoires ──
        missing_fields = []
        for field in self._mandatory_fields:
            value = extracted_data.get(field, "")
            if not value or len(str(value).strip()) < 2:
                missing_fields.append(field)

        if missing_fields:
            print(f"    [!] Champs obligatoires manquants : {', '.join(missing_fields)}")
            return False

        # ── Étape 2 (SECONDAIRE) : vérifier les labels détectés dans le layout ──
        for line in layout:
            for item in (line.get("items") or []):
                text = item.get("text", "")
                if not text:
                    continue
                field, _ = self.field_matcher.match_with_keyword(text)
                if field and field in self._main_field_names:
                    value = extracted_data.get(field, "")
                    if not value or len(str(value).strip()) < 2:
                        print(f"    [!] Label '{field}' détecté dans le layout mais valeur absente")
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
