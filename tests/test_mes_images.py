"""
Teste ImageQualityChecker sur toutes tes images réelles d'un dossier.
Utile pour calibrer les seuils (accept_threshold, warning_threshold,
critical_floor) selon la qualité réelle de tes documents.

Usage (depuis la racine du projet) :
    python tests/test_mes_images.py images
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.quality.image_quality_checker import ImageQualityChecker

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main(folder_path: str):
    checker = ImageQualityChecker()

    images = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"Aucune image trouvée dans {folder_path}")
        return

    print(f"{len(images)} image(s) trouvée(s)\n")
    print(f"{'Fichier':<25} {'Statut':<10} {'Score':<8} {'NettetéTexte':<14} {'ContrasteTexte':<16} {'Binarisation':<14} Raisons")
    print("-" * 120)

    for filename in sorted(images):
        path = os.path.join(folder_path, filename)
        try:
            r = checker.check(path)
            raisons = ", ".join(r["reasons"]) if r["reasons"] else "-"
            print(
                f"{filename:<25} {r['status']:<10} {r['score']:<8} "
                f"{r['metrics']['text_sharpness']['score']:<14} "
                f"{r['metrics']['local_text_contrast']['score']:<16} "
                f"{r['metrics']['binarization_quality']['score']:<14} {raisons}"
            )
        except Exception as e:
            print(f"{filename:<25} ERREUR: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python tests/test_mes_images.py images")
        sys.exit(1)

    main(sys.argv[1])