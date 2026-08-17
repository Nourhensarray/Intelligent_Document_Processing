import os
import csv
from pathlib import Path
from flask import Flask, render_template, request, flash, redirect, url_for, send_file, jsonify
import tkinter as tk
from tkinter import filedialog

from app.batch_processor import process_images_in_parallel_generator
from app.data_source import list_images_in_folder

app = Flask(__name__, template_folder="app/templates")
app.secret_key = "secret_ai_document_key_super_secure"

@app.route("/", methods=["GET"])
def index():
    return render_template("webapp.html", results=None)

@app.route("/browse", methods=["GET"])
def browse_folder():
    """Ouvre une fenêtre locale pour choisir un dossier et renvoie le chemin."""
    # Améliorer la qualité de la fenêtre sous Windows (DPI Awareness)
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # Création d'une fenêtre cachée
    root = tk.Tk()
    root.withdraw()
    # Met la fenêtre au premier plan
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Sélectionner le dossier d'images")
    root.destroy()
    
    return jsonify({"folder_path": folder_path})

@app.route("/process", methods=["POST"])
def process():
    source_type = request.form.get("source_type")
    
    if source_type == "folder":
        folder_path = request.form.get("folder_path")
        if not folder_path or not os.path.exists(folder_path):
            flash("Le dossier spécifié n'existe pas ou est invalide.", "error")
            return redirect(url_for("index"))
        
        images = list_images_in_folder(folder_path)
    else:
        flash("Seul le dossier d'images est supporté pour le moment.", "error")
        return redirect(url_for("index"))

    if not images:
        flash(f"Aucune image trouvée dans le dossier '{folder_path}'.", "error")
        return redirect(url_for("index"))

    # Récupérer l'option Mode Rapide
    fast_mode = request.form.get("fast_mode") == "1"

    # Lancement du traitement
    # Pour l'interface Web, le mode GPU séquentiel est idéal pour la stabilité
    success_list = []
    failed_list = []
    
    global stop_requested
    stop_requested = False
    
    for record in process_images_in_parallel_generator(images, use_gpu=True, fast_mode=fast_mode):
        if stop_requested:
            flash("Traitement interrompu par l'utilisateur.", "warning")
            break
            
        status = record.get("status", "UNKNOWN")
        if status == "SUCCESS":
            success_list.append(record)
        else:
            # Règle stricte pour les FAILED : on ne garde que l'image et le statut
            failed_list.append({
                "image": record.get("image", "Inconnu"),
                "status": status
            })
            
    # Écriture du CSV avec la séparation (SUCCESS en haut, FAILED en bas)
    results_list = success_list + failed_list
    csv_filename = "resultats_extraction.csv"
    csv_path = os.path.abspath(os.path.join(app.root_path, csv_filename))
    
    if results_list:
        logical_order = [
            "image", "status", "numero_document", "nom", "prenom", 
            "date_naissance", "lieu_naissance", "sexe", "nationalite", 
            "adresse", "date_delivrance", "date_expiration"
        ]
        
        all_keys = set()
        for r in success_list:
            all_keys.update(r.keys())
        all_keys.update({"image", "status"})
            
        fieldnames = []
        for col in logical_order:
            if col in all_keys:
                fieldnames.append(col)
                all_keys.remove(col)
        fieldnames.extend(sorted(list(all_keys)))
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for r in results_list:
                writer.writerow(r)
                
        flash(f"Traitement terminé avec succès. {len(success_list)} SUCCESS, {len(failed_list)} FAILED.", "success")
    
    return render_template("webapp.html", results=results_list, csv_filename=csv_filename)

@app.route("/download/<filename>")
def download(filename):
    # Sécuriser l'accès au fichier pour ne permettre que le téléchargement du CSV
    if filename != "resultats_extraction.csv":
        return "Accès non autorisé", 403
    csv_path = os.path.abspath(os.path.join(app.root_path, filename))
    if not os.path.exists(csv_path):
        return "Fichier non trouvé", 404
    return send_file(csv_path, as_attachment=True)

stop_requested = False

@app.route("/stop", methods=["POST"])
def stop():
    global stop_requested
    stop_requested = True
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    print("================================================================")
    print(" Lancement de l'interface Web - Serveur Local (Flask)")
    print(" Accédez à : http://127.0.0.1:5000 dans votre navigateur")
    print("================================================================")
    # Lancement sur le port 5000 (appuyez sur CTRL+C pour quitter)
    app.run(host="127.0.0.1", port=5000, debug=True)
