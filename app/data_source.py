import sqlite3
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


class UnsupportedDatabaseError(Exception):
    pass


def list_images_in_folder(folder_path: str) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Le dossier n'existe pas : {folder_path}")

    images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(str(p) for p in images)


def _parse_sqlite_path(database_url: str) -> Path:
    url = database_url.strip()
    if url.startswith("sqlite:///"):
        url = url[len("sqlite:///"):]
    elif url.startswith("sqlite://"):
        url = url[len("sqlite://"):]
    return Path(url)


def _normalize_image_path(db_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate

    relative_candidate = db_path.parent / candidate
    if relative_candidate.exists():
        return relative_candidate

    return candidate


def list_images_from_sqlite(database_url: str, query: str = "SELECT path FROM images") -> List[str]:
    db_path = _parse_sqlite_path(database_url)
    if not db_path.exists() or not db_path.is_file():
        raise FileNotFoundError(f"Fichier SQLite introuvable : {db_path}")

    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(query)
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Requête SQL invalide : {exc}") from exc

        rows = cursor.fetchall()

    if not rows:
        raise ValueError("Aucune ligne retournée par la requête SQL.")

    image_paths = []
    missing = []
    for row in rows:
        if not row:
            continue
        path_value = str(row[0]).strip()
        if not path_value:
            continue

        resolved = _normalize_image_path(db_path, path_value)
        if resolved.exists():
            image_paths.append(str(resolved))
        else:
            missing.append(path_value)

    if not image_paths:
        raise FileNotFoundError(
            "Aucune image valide trouvée dans la base de données. "
            "Vérifiez le champ path et l'emplacement des fichiers."
        )

    return sorted(image_paths)
