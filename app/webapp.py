import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from app.data_source import list_images_from_sqlite, list_images_in_folder
from app.pipeline import DocumentPipeline

OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def create_app():
    app = Flask(__name__)
    app.secret_key = "change-this-secret-for-production"
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
    app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("webapp.html", results=None, csv_filename=None)

    @app.route("/process", methods=["POST"])
    def process():
        source_type = request.form.get("source_type", "folder")
        images = []
        db_file_path = None

        try:
            if source_type == "folder":
                folder_path = request.form.get("folder_path", "").strip()
                if not folder_path:
                    raise ValueError("Veuillez renseigner le chemin du dossier d'images.")
                images = list_images_in_folder(folder_path)
            else:
                database_path = request.form.get("database_path", "").strip()
                database_query = request.form.get("database_query", "SELECT path FROM images").strip() or "SELECT path FROM images"
                sqlite_file = request.files.get("sqlite_file")

                if sqlite_file and sqlite_file.filename:
                    safe_path = UPLOAD_DIR / sqlite_file.filename
                    sqlite_file.save(str(safe_path))
                    db_file_path = str(safe_path)
                elif database_path:
                    db_file_path = database_path
                else:
                    raise ValueError("Veuillez renseigner un chemin SQLite ou uploader un fichier SQLite.")

                images = list_images_from_sqlite(db_file_path, database_query)

            if not images:
                raise ValueError("Aucune image trouvée pour le traitement.")

            from app.batch_processor import process_images_in_parallel_generator
            
            results = list(process_images_in_parallel_generator(images))

            output_file = OUTPUT_DIR / "resultats_extraction_web.csv"
            _save_results(output_file, results)
            csv_filename = output_file.name
            flash(f"Traitement terminé : {len(results)} image(s) traitée(s).", "success")
            return render_template("webapp.html", results=results, csv_filename=csv_filename)

        except Exception as exc:
            flash(str(exc), "error")
            return render_template("webapp.html", results=None, csv_filename=None)

    @app.route("/download/<path:filename>")
    def download(filename):
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

    return app


def _save_results(output_path: Path, results):
    fieldnames = ["image", "status"]
    for record in results:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
