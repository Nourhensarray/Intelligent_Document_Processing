import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    else:
        print("\n[X] Pipeline interrompu : L'image n'a pas franchi le contrôle de qualité.")


def main():
    pipeline = DocumentPipeline()

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        run_pipeline_on_image(pipeline, target_path)
    else:
        images_dir = Path("images")
        if images_dir.exists():
            image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if image_files:
                for img in sorted(image_files):
                    run_pipeline_on_image(pipeline, str(img))
            else:
                print("Aucune image trouvée dans le dossier images/")
        else:
            print("Usage : python tests/test_pipeline.py <chemin_image>")


if __name__ == "__main__":
    main()