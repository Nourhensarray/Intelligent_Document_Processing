import time
import os
import argparse
from pathlib import Path
from app.data_source import list_images_in_folder
from app.batch_processor import process_images_in_parallel_generator

def main():
    parser = argparse.ArgumentParser(description="Test du pipeline OCR optimisé en mode console.")
    parser.add_argument("--folder", type=str, default="images", help="Dossier contenant les images à traiter")
    parser.add_argument("--workers", type=int, default=None, help="Nombre de workers (mode CPU uniquement)")
    parser.add_argument("--cpu", action="store_true", help="Forcer le mode CPU multiprocessing (désactive le GPU)")
    args = parser.parse_args()

    folder = args.folder
    use_gpu = not args.cpu

    print(f"Recherche d'images dans le dossier : {folder}")
    workers_label = args.workers or "auto"
    mode_label = "GPU (séquentiel)" if use_gpu else f"CPU (multiprocessing, workers={workers_label})"
    print(f"Mode : {mode_label}")

    if not os.path.exists(folder):
        print(f"Le dossier '{folder}' n'existe pas.")
        return

    images = list_images_in_folder(folder)

    if not images:
        print(f"Aucune image trouvée dans '{folder}'.")
        return

    print(f"{len(images)} image(s) trouvée(s). Démarrage du traitement batch...")

    start_time = time.time()

    results = []
    for idx, record in enumerate(
        process_images_in_parallel_generator(images, num_workers=args.workers, use_gpu=use_gpu),
        start=1
    ):
        image_name = record.get("image", "Inconnu")
        status = record.get("status", "UNKNOWN")
        print(f"[{idx}/{len(images)}] Terminé : {image_name} (Status: {status})")
        results.append(record)

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "="*50)
    print("RÉSUMÉ DU TRAITEMENT")
    print("="*50)
    print(f"Mode            : {'GPU séquentiel' if use_gpu else 'CPU multiprocessing'}")
    print(f"Images traitées : {len(results)}")
    print(f"Temps total     : {elapsed:.2f} secondes")
    if len(results) > 0:
        print(f"Temps moyen     : {elapsed/len(results):.2f} sec / image")

    print("\nExemples de données extraites :")
    for r in results[:3]:
        print(r)

if __name__ == "__main__":
    main()
