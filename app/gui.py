"""
Interface graphique premium — AI Document Extractor
Dark theme avec cartes, barre de progression et résultats en temps réel.
"""
import csv
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from app.data_source import list_images_in_folder, list_images_from_sqlite, UnsupportedDatabaseError
from app.pipeline import DocumentPipeline

OUTPUT_CSV = Path("outputs/resultats_extraction_gui.csv")

# ──────────────────────────────────────────────
# PALETTE
# ──────────────────────────────────────────────
BG        = "#0f1117"   # fond principal
CARD      = "#1a1d2e"   # fond des cartes
BORDER    = "#2a2d3e"   # bordures subtiles
ACCENT    = "#00d4ff"   # cyan accent
ACCENT2   = "#7c3aed"   # violet secondaire
SUCCESS_C = "#22c55e"   # vert
WARN_C    = "#f59e0b"   # orange
ERROR_C   = "#ef4444"   # rouge
TEXT      = "#e2e8f0"   # texte principal
MUTED     = "#64748b"   # texte secondaire
BTN_BG    = "#1e40af"   # bouton principal
BTN_HOVER = "#2563eb"
BTN_RUN   = "#0891b2"
BTN_RUN_H = "#0e7490"


class RoundedCard(tk.Frame):
    """Frame simulant une carte avec fond CARD et bordure subtile."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1, **kwargs)


class ToggleButton(tk.Button):
    """Bouton toggle style pill."""
    def __init__(self, parent, text, variable, value, **kwargs):
        super().__init__(parent, text=text, relief=tk.FLAT, cursor="hand2",
                         font=("Segoe UI", 10, "bold"), bd=0,
                         padx=14, pady=6, **kwargs)
        self._var = variable
        self._value = value
        self._refresh()
        variable.trace_add("write", lambda *_: self._refresh())
        self.bind("<Button-1>", lambda _: variable.set(value))

    def _refresh(self):
        if self._var.get() == self._value:
            self.config(bg=ACCENT, fg="#0f1117")
        else:
            self.config(bg=BORDER, fg=MUTED)


class DocumentExtractorUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Document Extractor")
        self.root.geometry("980x720")
        self.root.minsize(860, 640)
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # Centrer la fenêtre
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - 980) // 2
        y = (self.root.winfo_screenheight() - 720) // 2
        self.root.geometry(f"980x720+{x}+{y}")

        self.log_queue: queue.Queue = queue.Queue()
        self.processing_thread = None
        self._total = 0
        self._done = 0
        self._success = 0
        self._failed = 0
        self._errors = 0

        self.source_type    = tk.StringVar(value="folder")
        self.folder_path    = tk.StringVar()
        self.database_url   = tk.StringVar()
        self.database_query = tk.StringVar(value="SELECT path FROM images")

        self._build_ui()
        self._schedule_log_flush()

    # ──────────────────────────────────────────
    # CONSTRUCTION UI
    # ──────────────────────────────────────────

    def _build_ui(self):
        # ── En-tête ──
        header = tk.Frame(self.root, bg=CARD, height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="⚡", font=("Segoe UI Emoji", 22), bg=CARD, fg=ACCENT).pack(side=tk.LEFT, padx=(20, 8), pady=12)
        title_f = tk.Frame(header, bg=CARD)
        title_f.pack(side=tk.LEFT, pady=12)
        tk.Label(title_f, text="AI Document Extractor", font=("Segoe UI", 16, "bold"),
                 bg=CARD, fg=TEXT).pack(anchor=tk.W)
        tk.Label(title_f, text="OCR  •  Extraction intelligente  •  Multi-format",
                 font=("Segoe UI", 8), bg=CARD, fg=MUTED).pack(anchor=tk.W)

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

        # ── Corps principal scrollable ──
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)

        # Colonne gauche (paramètres)
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self._build_source_card(left)
        self._build_action_card(left)
        self._build_counters(left)

        # Colonne droite (journal)
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_log_card(right)

        # ── Barre de statut ──
        self._build_statusbar()

        self.source_type.trace_add("write", lambda *_: self._on_source_change())
        self._on_source_change()

    def _build_source_card(self, parent):
        card = RoundedCard(parent, padx=16, pady=16)
        card.pack(fill=tk.X, pady=(0, 12))

        self._card_title(card, "📂  Source des images")

        # Toggle buttons
        toggle_f = tk.Frame(card, bg=CARD)
        toggle_f.pack(fill=tk.X, pady=(10, 14))
        ToggleButton(toggle_f, "📁  Dossier", self.source_type, "folder").pack(side=tk.LEFT, padx=(0, 8))
        ToggleButton(toggle_f, "🗄  SQLite",  self.source_type, "sqlite").pack(side=tk.LEFT)

        # Dossier
        self.folder_frame = tk.Frame(card, bg=CARD)
        self.folder_frame.pack(fill=tk.X)
        tk.Label(self.folder_frame, text="Chemin du dossier", font=("Segoe UI", 9),
                 bg=CARD, fg=MUTED).pack(anchor=tk.W)
        row = tk.Frame(self.folder_frame, bg=CARD)
        row.pack(fill=tk.X, pady=(4, 0))
        self.folder_entry = tk.Entry(row, textvariable=self.folder_path,
                                     bg=BG, fg=TEXT, insertbackground=ACCENT,
                                     relief=tk.FLAT, font=("Segoe UI", 10),
                                     highlightbackground=BORDER, highlightthickness=1)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self._btn(row, "Parcourir", self._select_folder, ACCENT2).pack(side=tk.LEFT, padx=(8, 0))

        # SQLite
        self.sqlite_frame = tk.Frame(card, bg=CARD)
        self.sqlite_frame.pack(fill=tk.X)
        tk.Label(self.sqlite_frame, text="Fichier SQLite", font=("Segoe UI", 9),
                 bg=CARD, fg=MUTED).pack(anchor=tk.W)
        row2 = tk.Frame(self.sqlite_frame, bg=CARD)
        row2.pack(fill=tk.X, pady=(4, 0))
        self.db_entry = tk.Entry(row2, textvariable=self.database_url,
                                 bg=BG, fg=TEXT, insertbackground=ACCENT,
                                 relief=tk.FLAT, font=("Segoe UI", 10),
                                 highlightbackground=BORDER, highlightthickness=1)
        self.db_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self._btn(row2, "Sélectionner", self._select_database, ACCENT2).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(self.sqlite_frame, text="Requête SQL", font=("Segoe UI", 9),
                 bg=CARD, fg=MUTED).pack(anchor=tk.W, pady=(10, 0))
        self.query_entry = tk.Entry(self.sqlite_frame, textvariable=self.database_query,
                                    bg=BG, fg=TEXT, insertbackground=ACCENT,
                                    relief=tk.FLAT, font=("Segoe UI Mono", 9),
                                    highlightbackground=BORDER, highlightthickness=1)
        self.query_entry.pack(fill=tk.X, ipady=6, pady=(4, 0))

    def _build_action_card(self, parent):
        card = RoundedCard(parent, padx=16, pady=14)
        card.pack(fill=tk.X, pady=(0, 12))

        self._card_title(card, "⚙️  Traitement")

        btn_row = tk.Frame(card, bg=CARD)
        btn_row.pack(fill=tk.X, pady=(12, 0))

        self.run_btn = tk.Button(
            btn_row, text="▶  Démarrer", font=("Segoe UI", 11, "bold"),
            bg=BTN_RUN, fg="white", relief=tk.FLAT, cursor="hand2",
            padx=20, pady=10, bd=0,
            command=self._on_run,
            activebackground=BTN_RUN_H, activeforeground="white"
        )
        self.run_btn.pack(side=tk.LEFT)
        self.run_btn.bind("<Enter>", lambda _: self.run_btn.config(bg=BTN_RUN_H))
        self.run_btn.bind("<Leave>", lambda _: self.run_btn.config(bg=BTN_RUN))

        self.stop_btn = tk.Button(
            btn_row, text="⏹  Arrêter", font=("Segoe UI", 11),
            bg=BORDER, fg=MUTED, relief=tk.FLAT, cursor="hand2",
            padx=16, pady=10, bd=0, state=tk.DISABLED,
            command=self._on_stop,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Barre de progression
        prog_f = tk.Frame(card, bg=CARD)
        prog_f.pack(fill=tk.X, pady=(14, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("DocScan.Horizontal.TProgressbar",
                        troughcolor=BG, background=ACCENT,
                        borderwidth=0, thickness=6)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            prog_f, variable=self.progress_var,
            maximum=100, style="DocScan.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = tk.Label(prog_f, text="En attente...", font=("Segoe UI", 8),
                                       bg=CARD, fg=MUTED)
        self.progress_label.pack(anchor=tk.W, pady=(4, 0))

    def _build_counters(self, parent):
        card = RoundedCard(parent, padx=16, pady=14)
        card.pack(fill=tk.X)

        self._card_title(card, "📊  Statistiques")

        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill=tk.X, pady=(12, 0))
        grid.columnconfigure((0, 1, 2, 3), weight=1)

        self._counter_box = {}
        items = [
            ("total",   "Total",   TEXT,     "—"),
            ("success", "SUCCESS", SUCCESS_C, "0"),
            ("failed",  "FAILED",  ERROR_C,   "0"),
            ("errors",  "ERREURS", WARN_C,    "0"),
        ]
        for col, (key, label, color, init) in enumerate(items):
            f = tk.Frame(grid, bg=BG, padx=10, pady=8,
                         highlightbackground=BORDER, highlightthickness=1)
            f.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            val_lbl = tk.Label(f, text=init, font=("Segoe UI", 20, "bold"),
                               bg=BG, fg=color)
            val_lbl.pack()
            tk.Label(f, text=label, font=("Segoe UI", 8),
                     bg=BG, fg=MUTED).pack()
            self._counter_box[key] = val_lbl

    def _build_log_card(self, parent):
        card = RoundedCard(parent, padx=16, pady=14)
        card.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(card, bg=CARD)
        hdr.pack(fill=tk.X)
        self._card_title(hdr, "📋  Journal en temps réel")
        self.clear_btn = tk.Button(hdr, text="Effacer", font=("Segoe UI", 8),
                                   bg=BORDER, fg=MUTED, relief=tk.FLAT, cursor="hand2",
                                   padx=8, pady=3, bd=0, command=self._clear_log)
        self.clear_btn.pack(side=tk.RIGHT)

        # Zone de log avec scrollbar custom
        log_frame = tk.Frame(card, bg=BG,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        scrollbar = tk.Scrollbar(log_frame, bg=CARD, troughcolor=BG,
                                  relief=tk.FLAT, width=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame,
            bg=BG, fg=TEXT,
            font=("Cascadia Code", 9) if self._font_exists("Cascadia Code") else ("Courier New", 9),
            relief=tk.FLAT, bd=0,
            state=tk.DISABLED,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            cursor="arrow",
            selectbackground=BORDER,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        scrollbar.config(command=self.log_text.yview)

        # Tags couleur pour le log
        self.log_text.tag_configure("success", foreground=SUCCESS_C)
        self.log_text.tag_configure("failed",  foreground=ERROR_C)
        self.log_text.tag_configure("error",   foreground=WARN_C)
        self.log_text.tag_configure("info",    foreground=ACCENT)
        self.log_text.tag_configure("muted",   foreground=MUTED)
        self.log_text.tag_configure("bold",    foreground=TEXT, font=("Segoe UI", 9, "bold"))

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=CARD, height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(bar, text="●  Prêt", font=("Segoe UI", 9),
                                      bg=CARD, fg=SUCCESS_C)
        self.status_label.pack(side=tk.LEFT, padx=14)

        self.time_label = tk.Label(bar, text="", font=("Segoe UI", 9),
                                    bg=CARD, fg=MUTED)
        self.time_label.pack(side=tk.RIGHT, padx=14)

    # ──────────────────────────────────────────
    # HELPERS UI
    # ──────────────────────────────────────────

    @staticmethod
    def _card_title(parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                 bg=CARD if isinstance(parent, tk.Frame) else CARD,
                 fg=TEXT).pack(anchor=tk.W)

    @staticmethod
    def _btn(parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd, font=("Segoe UI", 9),
                      bg=color, fg="white", relief=tk.FLAT, cursor="hand2",
                      padx=12, pady=6, bd=0,
                      activebackground=color, activeforeground="white")
        return b

    @staticmethod
    def _font_exists(name):
        try:
            import tkinter.font as tkfont
            return name in tkfont.families()
        except Exception:
            return False

    # ──────────────────────────────────────────
    # SOURCE TOGGLE
    # ──────────────────────────────────────────

    def _on_source_change(self):
        if self.source_type.get() == "folder":
            self.folder_frame.pack(fill=tk.X)
            self.sqlite_frame.pack_forget()
        else:
            self.folder_frame.pack_forget()
            self.sqlite_frame.pack(fill=tk.X)

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier d'images")
        if folder:
            self.folder_path.set(folder)

    def _select_database(self):
        db_file = filedialog.askopenfilename(
            title="Sélectionner le fichier SQLite",
            filetypes=[("SQLite Database", "*.db *.sqlite"), ("Tous les fichiers", "*")]
        )
        if db_file:
            self.database_url.set(db_file)

    # ──────────────────────────────────────────
    # LOG
    # ──────────────────────────────────────────

    def _schedule_log_flush(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self._append_to_log(*item)
        except queue.Empty:
            pass
        self.root.after(80, self._schedule_log_flush)

    def _append_to_log(self, message: str, tag: str = ""):
        self.log_text.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] ", "muted")
        self.log_text.insert(tk.END, f"{message}\n", tag if tag else "")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def log(self, message: str, tag: str = ""):
        self.log_queue.put((message, tag))

    # ──────────────────────────────────────────
    # STATUS & COUNTERS
    # ──────────────────────────────────────────

    def _set_status(self, message: str, color: str = TEXT):
        self.root.after(0, lambda: self.status_label.config(text=f"●  {message}", fg=color))

    def _update_counters(self):
        def _do():
            self._counter_box["total"].config(text=str(self._total) if self._total else "—")
            self._counter_box["success"].config(text=str(self._success))
            self._counter_box["failed"].config(text=str(self._failed))
            self._counter_box["errors"].config(text=str(self._errors))
            if self._total:
                pct = (self._done / self._total) * 100
                self.progress_var.set(pct)
                self.progress_label.config(
                    text=f"{self._done}/{self._total} images traitées  ({pct:.0f}%)"
                )
        self.root.after(0, _do)

    def _enable_controls(self, enabled: bool = True):
        def _do():
            if enabled:
                self.run_btn.config(state=tk.NORMAL, bg=BTN_RUN)
                self.stop_btn.config(state=tk.DISABLED, bg=BORDER, fg=MUTED)
            else:
                self.run_btn.config(state=tk.DISABLED, bg=BORDER)
                self.stop_btn.config(state=tk.NORMAL, bg="#7f1d1d", fg="white")
        self.root.after(0, _do)

    # ──────────────────────────────────────────
    # TRAITEMENT
    # ──────────────────────────────────────────

    def _on_run(self):
        if self.processing_thread and self.processing_thread.is_alive():
            return

        # Réinitialiser les compteurs
        self._total = self._done = self._success = self._failed = self._errors = 0
        self.progress_var.set(0)
        self.progress_label.config(text="Initialisation...")
        self._update_counters()
        self._start_time = time.time()

        try:
            if self.source_type.get() == "folder":
                folder = self.folder_path.get().strip()
                if not folder:
                    raise ValueError("Veuillez sélectionner un dossier d'images.")
                images = list_images_in_folder(folder)
            else:
                db_url = self.database_url.get().strip()
                query = self.database_query.get().strip() or "SELECT path FROM images"
                if not db_url:
                    raise ValueError("Veuillez renseigner le chemin du fichier SQLite.")
                images = list_images_from_sqlite(db_url, query)

            if not images:
                raise ValueError("Aucune image trouvée dans la source sélectionnée.")

            self._total = len(images)
            self._update_counters()
            self._enable_controls(False)
            self._set_status("Traitement en cours...", WARN_C)
            self.log(f"{'─'*50}", "muted")
            self.log(f"Démarrage : {self._total} image(s) à traiter", "info")
            self.log(f"{'─'*50}", "muted")

            self._stop_flag = threading.Event()
            self.processing_thread = threading.Thread(
                target=self._process_images, args=(images,), daemon=True
            )
            self.processing_thread.start()
            self._update_timer()

        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            self.log(f"Erreur : {exc}", "error")

    def _on_stop(self):
        if hasattr(self, "_stop_flag"):
            self._stop_flag.set()
            self._set_status("Arrêt demandé...", WARN_C)
            self.log("Arrêt demandé par l'utilisateur", "error")

    def _update_timer(self):
        if self.processing_thread and self.processing_thread.is_alive():
            elapsed = time.time() - self._start_time
            self.time_label.config(text=f"⏱  {elapsed:.0f}s")
            self.root.after(1000, self._update_timer)
        else:
            if hasattr(self, "_start_time"):
                elapsed = time.time() - self._start_time
                self.time_label.config(text=f"⏱  {elapsed:.1f}s")

    def _process_images(self, images):
        results = []
        try:
            from app.batch_processor import process_images_in_parallel_generator

            for index, record in enumerate(process_images_in_parallel_generator(images), start=1):
                if hasattr(self, "_stop_flag") and self._stop_flag.is_set():
                    self.log("⏹  Traitement interrompu.", "error")
                    break

                image_name = record.get("image", "Inconnu")
                status = record.get("status", "UNKNOWN")
                self._done += 1

                if status == "SUCCESS":
                    self._success += 1
                    tag = "success"
                    icon = "✓"
                elif "ERROR" in status:
                    self._errors += 1
                    tag = "error"
                    icon = "✗"
                else:
                    self._failed += 1
                    tag = "failed"
                    icon = "✗"

                self.log(f"[{index:3}/{self._total}] {icon} {image_name}  →  {status}", tag)
                self._update_counters()
                results.append(record)

            if results:
                output_path = self._save_results(results)
                self.log(f"{'─'*50}", "muted")
                self.log(f"✔  Résultats sauvegardés : {output_path}", "success")
                self.log(f"   SUCCESS: {self._success}  |  FAILED: {self._failed}  |  ERREURS: {self._errors}", "info")

            self._set_status("Terminé.", SUCCESS_C)

        except Exception as exc:
            self.log(f"Erreur critique : {exc}", "error")
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(exc)))
            self._set_status("Erreur.", ERROR_C)
        finally:
            self._enable_controls(True)
            self.progress_var.set(100 if self._done == self._total else self.progress_var.get())

    def _save_results(self, results):
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        all_keys: set = set()
        for record in results:
            all_keys.update(record.keys())

        fieldnames = ["image", "status"] + sorted(k for k in all_keys if k not in {"image", "status"})

        with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for record in results:
                writer.writerow(record)

        return OUTPUT_CSV


def main():
    root = tk.Tk()
    try:
        # Icône (ignoré si introuvable)
        root.iconbitmap(default="")
    except Exception:
        pass
    DocumentExtractorUI(root)
    root.mainloop()
