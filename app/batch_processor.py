import multiprocessing as mp
import traceback
from pathlib import Path

from app.pipeline import DocumentPipeline

# Pipeline global pour le mode multiprocessing CPU.
_global_pipeline = None


def _init_worker():
    """
    Initialise le pipeline OCR une seule fois par worker CPU.
    """
    global _global_pipeline
    try:
        _global_pipeline = DocumentPipeline(fast_mode=False)
    except Exception as e:
        print(f"Erreur d'initialisation du worker OCR: {e}")
        traceback.print_exc()


def _worker_task(image_path: str) -> dict:
    """
    Traite une image en utilisant le pipeline OCR global (mode CPU multiprocessing).
    """
    global _global_pipeline
    if _global_pipeline is None:
        return {"image": Path(image_path).name, "status": "ERROR: Worker non initialisé"}
    try:
        result = _global_pipeline.process(image_path)
        record = {"image": Path(image_path).name, "status": result.get("status", "UNKNOWN")}
        if result.get("data"):
            record.update(result["data"])
        return record
    except Exception as e:
        return {"image": Path(image_path).name, "status": f"ERROR: {str(e)}"}


def process_images_in_parallel_generator(images: list, num_workers: int = None, use_gpu: bool = True):
    """
    Traite une liste d'images et renvoie (yield) les résultats au fur et à mesure.

    - Mode GPU (use_gpu=True) : séquentiel avec une seule instance PaddleOCR.
      PaddlePaddle GPU n'est pas compatible avec le multiprocessing "spawn" de
      Windows. La vitesse GPU vient du traitement en batch (rec_batch_num=32),
      pas du nombre de processus.

    - Mode CPU (use_gpu=False) : pool de workers en multiprocessing pour
      exploiter tous les cœurs CPU.

    Args:
        images     : Liste de chemins vers les images.
        num_workers: Ignoré en mode GPU. En mode CPU : nombre de workers
                     (défaut : cpu_count - 1, max 6).
        use_gpu    : True pour le mode GPU séquentiel, False pour le pool CPU.
    """
    if use_gpu:
        # ── Mode GPU : 1 seul pipeline, inférence rapide grâce au GPU ──
        print("Mode GPU : initialisation unique du moteur OCR...")
        pipeline = DocumentPipeline(fast_mode=False)
        for image_path in images:
            try:
                result = pipeline.process(image_path)
                record = {"image": Path(image_path).name, "status": result.get("status", "UNKNOWN")}
                if result.get("data"):
                    record.update(result["data"])
            except Exception as e:
                record = {"image": Path(image_path).name, "status": f"ERROR: {str(e)}"}
            yield record
    else:
        # ── Mode CPU : pool de workers en multiprocessing ──
        if num_workers is None:
            cpu_count = mp.cpu_count()
            num_workers = max(1, min(6, cpu_count - 1))
        print(f"Mode CPU : pool de {num_workers} workers...")
        with mp.Pool(processes=num_workers, initializer=_init_worker) as pool:
            for record in pool.imap_unordered(_worker_task, images):
                yield record

