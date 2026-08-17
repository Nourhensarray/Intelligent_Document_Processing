import sys
import os
import csv
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.extraction.field_matcher import FieldMatcher
from app.extraction.value_extractor import ValueExtractor
from app.pipeline import DocumentPipeline


def run_pipeline_on_image(pipeline: DocumentPipeline, image_path: str):
    print("\n" + "=" * 60)
    print(f"TEST DU PIPELINE SUR : {image_path}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"[X] Erreur : Fichier non trouvé : {image_path}")
        return

    result = pipeline.process(image_path)

    print("\n--- STATUT PIPELINE ---")
    print(f"Statut : {result['status']}")
    print(f"Score Qualité : {result['quality']['score']}/100 ({result['quality']['status']})")

    if result['quality']['reasons']:
        print("Raisons du contrôle qualité :")
        for r in result['quality']['reasons']:
            print(f"  - {r}")

    if result["status"] == "SUCCESS":
        print("\n--- ZONES OCR DETECTEES ---")
        if result["ocr"]:
            for item in result["ocr"]:
                print(f"  • {repr(item['text']):<35} (Confiance: {item['confidence']:.2f})")
        else:
            print("  Aucune zone OCR trouvée.")

        print("\n--- DONNEES EXTRAITES (CLÉ-VALEUR) ---")
        print(result["data"])
        return {"image": os.path.basename(image_path), "status": "SUCCESS", **result["data"]}
    else:
        print("\n[X] Pipeline interrompu : L'image n'a pas franchi le contrôle de qualité.")
        return {"image": os.path.basename(image_path), "status": "FAILED"}


def main():
    pipeline = DocumentPipeline()
    success_list = []
    failed_list = []

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        res = run_pipeline_on_image(pipeline, target_path)
        if res:
            if res.get("status") == "SUCCESS":
                success_list.append(res)
            else:
                failed_list.append(res)
    else:
        images_dir = Path("images")
        if images_dir.exists():
            image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if image_files:
                for img in sorted(image_files):
                    res = run_pipeline_on_image(pipeline, str(img))
                    if res:
                        if res.get("status") == "SUCCESS":
                            success_list.append(res)
                        else:
                            failed_list.append(res)
            else:
                print("Aucune image trouvée dans le dossier images/")
        else:
            print("Usage : python tests/test_pipeline.py <chemin_image>")

    # Écrire le CSV : SUCCESS en haut, FAILED en bas (sans valeurs)
    results_list = success_list + failed_list

    if results_list:
        csv_path = "resultats_extraction.csv"
        
        # Ordre logique des colonnes (les champs connus d'abord)
        logical_order = [
            "image", "status", "numero_document", "nom", "prenom", 
            "date_naissance", "lieu_naissance", "sexe", "nationalite", 
            "adresse", "date_delivrance", "date_expiration"
        ]
        
        # Collecter toutes les clés trouvées (uniquement depuis les SUCCESS)
        all_keys = set()
        for r in success_list:
            all_keys.update(r.keys())
        # S'assurer que image et status sont toujours présents
        all_keys.update({"image", "status"})
            
        # Créer les fieldnames : l'ordre logique d'abord, puis les autres clés imprévues
        fieldnames = []
        for col in logical_order:
            if col in all_keys:
                fieldnames.append(col)
                all_keys.remove(col)
        
        # Ajouter le reste (s'il y en a) par ordre alphabétique
        fieldnames.extend(sorted(list(all_keys)))
        
        # utf-8-sig permet à Excel de lire les accents directement sans problème
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # delimiter=';' est requis pour l'Excel français
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for r in results_list:
                writer.writerow(r)
        print(f"\n[OK] Résultats sauvegardés dans {csv_path}")
        print(f"     → {len(success_list)} SUCCESS, {len(failed_list)} FAILED")

def test_pipeline_rejects_main_label_without_value():
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline.quality_checker = MagicMock()
    pipeline.preprocessor = MagicMock()
    pipeline.ocr_engine = MagicMock()
    pipeline.postprocessor = MagicMock()
    pipeline.layout_builder = MagicMock()
    pipeline.value_extractor = MagicMock()
    pipeline.field_matcher = FieldMatcher()
    pipeline._mandatory_fields = {"nom", "prenom", "numero_document", "nationalite"}
    pipeline._main_field_names = pipeline._mandatory_fields | {"date_naissance"}

    pipeline.quality_checker.check.return_value = {
        "status": "ACCEPTED",
        "score": 85,
        "metrics": {},
        "reasons": [],
    }
    pipeline.preprocessor.preprocess.return_value = "dummy_image"
    pipeline.ocr_engine.extract.return_value = [
        {"text": "nom", "confidence": 0.95, "box": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        {"text": "prenom", "confidence": 0.93, "box": [[0, 30], [120, 30], [120, 50], [0, 50]]},
    ]
    pipeline.postprocessor.process.side_effect = lambda data: data
    pipeline.layout_builder.build.return_value = [
        {"text": "nom prenom", "items": [
            {"text": "nom", "box": [[0, 0], [100, 0], [100, 20], [0, 20]]},
            {"text": "prenom", "box": [[0, 30], [120, 30], [120, 50], [0, 50]]},
        ]}
    ]
    pipeline.value_extractor.extract.return_value = {}

    result = pipeline.process("images/test.jpeg")

    assert result["status"] == "FAILED"
    assert result["failure_code"] == "REJECTED_UNCLEAR_KEYS"
    assert result["layout"] is not None
    assert result["ocr"] is not None

if __name__ == "__main__":
    main()