import re
import unicodedata


from app.extraction.field_matcher import FieldMatcher


class ValueExtractor:

    def __init__(self):

        self.matcher = FieldMatcher()

        # Distance maximale entre un label et sa valeur (en pixels)
        self.max_vertical_distance = 120
        self.max_horizontal_distance = 400

        # Tolérance pour fusionner des boxes sur la même ligne
        self.merge_horizontal_gap = 60

        # Patterns de validation par champ
        self._validators = {
            "nom":              re.compile(r"^[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ\-\s']{2,30}$"),
            "prenom":           re.compile(r"^[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ][a-zA-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇàâäéèêëîïôùûüç,\-\s']{1,40}$"),
            "nationalite":      re.compile(r"^(FRANCAISE|FRANÇAISE|FRA|FRENCH)$", re.IGNORECASE),
            "date_naissance":   re.compile(r"^\d{1,2}[\.\-\/]\d{1,2}[\.\-\/]\d{2,4}$|^\d{8}$"),
            "date_expiration":  re.compile(r"^\d{1,2}[\.\-\/]\d{1,2}[\.\-\/]\d{2,4}$|^\d{8}$"),
            "numero_document":  re.compile(r"^[A-Z0-9]{6,15}$"),
        }

    # =====================================================
    # POINT D'ENTRÉE
    # =====================================================

    def extract(self, ocr_data):

        result = {}

        if not ocr_data:
            return result

        items = list(ocr_data)

        # Étape 1 : MRZ (données les plus fiables)
        self._extract_from_mrz(items, result)

        # Étape 2 : Fusion des boxes sur la même ligne
        items = self.merge_same_line_items(items)

        # Étape 3 : Tri de haut en bas puis gauche à droite
        items = sorted(
            items,
            key=lambda item: (self.center_y(item["box"]), self.center_x(item["box"]))
        )

        # Étape 4 : Extraction label → valeur
        for i, label_item in enumerate(items):

            label_text = label_item["text"]
            
            # --- TENTATIVE DE RÉCUPÉRATION GLUÉE ---
            # Si le texte est collé (ex: "NCmMICHEL", "SexeF", "Nec)l08/08/1990"),
            # le matcher classique échouera. On essaie d'abord par regex.
            glued_field, glued_value = self._extract_glued_inline(label_text)
            if glued_field and self._is_valid(glued_field, glued_value):
                if glued_field not in result:
                    result[glued_field] = glued_value
                continue

            field, keyword = self.matcher.match_with_keyword(label_text)

            if field is None:
                continue

            # Ne pas écraser une valeur MRZ ou gluée fiable
            if field in result and result[field]:
                continue

            # Cas inline classique (séparé par espace ou ':' mais capté par le matcher)
            inline_value = self.extract_inline_value(label_text, keyword)
            if inline_value and self._is_valid(field, inline_value):
                result[field] = inline_value
                continue

            # Recherche spatiale
            value = self.find_value(label_item, items, i)
            if value is not None and self._is_valid(field, value):
                result[field] = value

        # Étape 5 : Fallbacks simples pour les champs encore manquants
        self._apply_fallbacks(items, result)

        # Étape 6 : Nettoyage final
        for k in list(result.keys()):
            val = str(result[k]).strip(" :.-\u00a0/")
            if not val or (len(val) < 2 and k != "sexe"):
                del result[k]
            else:
                result[k] = val

        return result


    # =====================================================
    # PRIORITÉ 1 : EXTRACTION DEPUIS LA MRZ
    # =====================================================

    def _extract_from_mrz(self, items, result):
        """
        Parse les lignes MRZ (Machine Readable Zone) pour extraire les données clés.
        Format passeport (TD3) : 2 lignes de 44 chars
        Format CNI (TD1)       : 3 lignes de 30 chars
        """
        mrz_lines = []
        for item in items:
            t = item["text"].strip()
            # Une ligne MRZ contient forcément des "<"
            if "<" in t and len(t) >= 15:
                mrz_lines.append(t)

        if not mrz_lines:
            return

        for line in mrz_lines:
            # Ligne de nom de passeport : P<FRA BERTHIER<<CORINNE...
            if line.startswith("P<") or line.startswith("P<FRA"):
                self._parse_mrz_name_line(line, result)

            # Ligne de nom de CNI : IDFRABERTHIER<<...
            elif re.match(r"^(IDFRA|I<FRA|ID[A-Z]{3})", line):
                self._parse_mrz_name_line(line, result)

            # Ligne de données (numéro document, date naissance, date expiration)
            else:
                self._parse_mrz_data_line(line, result)

    def _parse_mrz_name_line(self, line, result):
        """Extrait nom + prénom d'une ligne MRZ de type nom."""
        # Supprimer seulement le préfixe de document connu (P<FRA, IDFRA, I<FRA)
        # en laissant intact le nom qui suit
        clean = re.sub(r"^(?:P<[A-Z]{3}|ID[A-Z]{3}|I<[A-Z]{3})", "", line)
        parts = clean.split("<<", 1)

        if parts[0] and "nom" not in result:
            surname = parts[0].replace("<", " ").strip()
            if surname and re.match(r"^[A-Z\- ]{2,30}$", surname):
                result["nom"] = surname

        if len(parts) >= 2 and parts[1] and "prenom" not in result:
            given = re.sub(r"<+", " ", parts[1]).strip()
            if given:
                result["prenom"] = given.title()

    def _parse_mrz_data_line(self, line, result):
        """
        Extrait numéro document, dates depuis la 2e ligne MRZ.
        Supporte TD3 (passeport, 44 chars) et TD1 (CNI, 30 chars).
        """
        clean = line.replace(" ", "")
        length = len(clean)

        # ---- TD3 (passeport) : 44 chars exactement ----
        # pos 1-9  : numéro document
        # pos 10   : chiffre de contrôle
        # pos 11-13: nationalité
        # pos 14-19: date naissance (YYMMDD)
        # pos 20   : chiffre de contrôle
        # pos 21   : sexe (M/F/<)
        # pos 22-27: date expiration (YYMMDD)
        if length >= 40:
            # Numéro document : 9 premiers chars, strip '<'
            doc = clean[:9].rstrip("<")
            if doc and re.match(r"^[A-Z0-9]{3,9}$", doc) and "numero_document" not in result:
                result["numero_document"] = doc

            # Dates uniquement si le marqueur de sexe (M/F) est à la position 20
            sex_marker = clean[20:21] if length > 20 else ""
            if sex_marker in ("M", "F"):
                dob_str = clean[13:19]   # positions 14-19 (index 13-18)
                exp_str = clean[21:27]   # positions 22-27 (index 21-26)
                if re.match(r"^\d{6}$", dob_str) and "date_naissance" not in result:
                    result["date_naissance"] = self._format_mrz_date(dob_str)
                if re.match(r"^\d{6}$", exp_str) and "date_expiration" not in result:
                    result["date_expiration"] = self._format_mrz_date(exp_str)
            return

        # ---- TD1 (CNI) : ~30 chars ----
        # Numéro document en début de ligne
        if length >= 15:
            match_doc = re.match(r"^([A-Z0-9]{6,9})", clean)
            if match_doc and "numero_document" not in result:
                doc = match_doc.group(1).rstrip("<")
                if len(doc) >= 6:
                    result["numero_document"] = doc

            # Chercher sexe pour séparer les dates
            td1_match = re.search(r"(\d{6})\d?[MF<](\d{6})", clean)
            if td1_match:
                if "date_naissance" not in result:
                    result["date_naissance"] = self._format_mrz_date(td1_match.group(1))
                if "date_expiration" not in result:
                    result["date_expiration"] = self._format_mrz_date(td1_match.group(2))

    @staticmethod
    def _format_mrz_date(d):
        """Convertit YYMMDD → DD.MM.YYYY (estimé)."""
        if len(d) != 6:
            return d
        yy, mm, dd = d[:2], d[2:4], d[4:]
        # Hypothèse : si yy > 30 → 19xx, sinon 20xx
        year = f"19{yy}" if int(yy) > 30 else f"20{yy}"
        return f"{dd}.{mm}.{year}"


    # =====================================================
    # EXTRACTION INLINE (label + valeur dans la même box)
    # =====================================================

    def extract_inline_value(self, label_text, keyword):
        """
        Si le texte OCR contient à la fois le label et sa valeur
        (ex: "Nom: BERTHIER" ou "Vom:BERTHIER"), isole la valeur.
        """
        if not keyword:
            return None

        # Libellé bilingue (ex: "NOM/Smam") → la partie après "/" est sous-titre, pas une valeur
        if "/" in label_text:
            parts = label_text.split("/", 1)
            remainder = parts[1].strip()
            if len(remainder) <= 8 or remainder.lower() in ["smam", "surname", "nationality", "sex", "a"]:
                return None

        normalized = self._strip_accents(label_text.lower())
        kw_norm = self._strip_accents(keyword.lower())

        idx = normalized.find(kw_norm)

        if idx == -1:
            # Typo OCR sur le mot clé (ex: "Vom:" pour "Nom:")
            match_typo = re.search(r"^[a-z]{2,6}\s*:\s*", normalized)
            if match_typo:
                remainder = label_text[match_typo.end():].strip(" :.-\u00a0")
                if remainder and len(remainder) >= 2:
                    return remainder
            return None

        remainder = label_text[idx + len(keyword):].strip(" :.-\u00a0")

        if not remainder or len(remainder) < 2:
            return None

        # Vérifier que ce n'est pas un autre label
        other_field, _ = self.matcher.match_with_keyword(remainder)
        if other_field is not None:
            return None

        return remainder


    def _extract_glued_inline(self, text):
        """
        Gère les cas où l'OCR colle le label et la valeur avec des erreurs.
        Ex: 'NCmMICHEL' -> Nom: MICHEL
        Ex: 'SexeF' -> Sexe: F
        Ex: 'Nec)l08/08/1990' -> date_naissance: 08/08/1990
        """
        # Nettoyage brutal des espaces et mise en minuscules pour faciliter les regex
        clean = text.lower().replace(" ", "").replace(":", "")

        # 1. Nom
        # Cherche "nom", "vom", "ncm", "nam"
        m = re.match(r"^(nom|vom|ncm|nam)([a-zà-ÿ\-\']{2,30})$", clean)
        if m:
            return "nom", text[len(m.group(1)):].strip(" :.-\u00a0")
            
        # 2. Prénom
        # Cherche "prenom", "prenoms", "prnoms", "prehos"
        m = re.match(r"^(prenoms?|prnoms?|prehos|firstnames?)[)\]]*([a-zà-ÿ\,\-\']{2,40})$", clean)
        if m:
            return "prenom", text[len(m.group(1)):].lstrip(")] :.-\u00a0").replace(",", ", ")

        # 3. Sexe
        m = re.match(r"^(sexe?|sex)([mf])$", clean)
        if m:
            return "sexe", m.group(2).upper()
            
        # 4. Date de naissance
        # Cherche "ne(e)le", "necle", "datedenaissance", "datedenaisso"
        m = re.match(r"^(ne\(?e?\)?l?e?|nec\)?l?e?|datedenaiss\w*|dob)([0-9\.\/\-]{8,10})$", clean)
        if m:
            return "date_naissance", m.group(2)
            
        # 5. Date d'expiration
        # Cherche "datedexpiration", "dateexpration", ignore les apostrophes éventuelles
        clean_no_quote = clean.replace("'", "").replace('"', "")
        m = re.match(r"^(dated?expir\w*|expir\w*)([0-9\.\/\-]{8,10})$", clean_no_quote)
        if m:
            return "date_expiration", m.group(2)
            
        # 6. Date d'émission
        m = re.match(r"^(dated?emiss\w*|datedelivr\w*)([0-9\.\/\-]{6,10})$", clean_no_quote)
        if m:
            return "date_delivrance", m.group(2)
            
        # 7. Numéro de document
        # Cherche "cartenationale...", "numerodocument" (non glouton)
        m = re.match(r"^(cartenationaled?ident[a-z]*|numerodocument|ndudocument|passportnumber)([a-z0-9]{6,15})$", clean_no_quote)
        if m:
            return "numero_document", m.group(2).upper()

        return None, None

    @staticmethod
    def _strip_accents(text):
        text = unicodedata.normalize("NFD", text)
        return "".join(c for c in text if unicodedata.category(c) != "Mn")


    # =====================================================
    # VALIDATION PAR CHAMP
    # =====================================================

    def _is_valid(self, field, value):
        """Vérifie que la valeur extraite est cohérente avec le champ attendu."""
        if not value or len(value.strip()) < 2:
            return False
        pattern = self._validators.get(field)
        if pattern is None:
            return True  # Pas de contrainte → accepter
        return bool(pattern.match(value.strip()))


    # =====================================================
    # RECHERCHE SPATIALE DE LA VALEUR
    # =====================================================

    def find_value(self, label_item, items, label_index):

        label_box = label_item["box"]
        label_y = self.center_y(label_box)
        label_right = label_box[1][0]
        label_x = self.center_x(label_box)

        candidates = []

        for i, item in enumerate(items):

            if i == label_index:
                continue

            text = item["text"].strip()
            box = item["box"]

            if not text or len(text) < 2:
                continue

            # Ignorer les sous-titres et bruits connus
            if text.lower() in ["smam", "surname", "nationality", "sex", "a", "m", "f"]:
                continue

            # Ignorer les autres labels
            other_field, _ = self.matcher.match_with_keyword(text)
            if other_field is not None:
                continue

            x = self.center_x(box)
            y = self.center_y(box)
            left = box[0][0]

            # Priorité 1 : même ligne, à droite du label
            if self.is_same_line(label_box, box) and left >= label_right - 10:
                h_dist = left - label_right
                if h_dist <= self.max_horizontal_distance:
                    candidates.append((0, h_dist, text))
                    continue

            # Priorité 2 : juste en dessous du label (verticalement proche)
            if y > label_y:
                v_dist = y - label_y
                h_dist = abs(x - label_x)
                if v_dist <= self.max_vertical_distance and h_dist <= self.max_horizontal_distance:
                    candidates.append((v_dist, h_dist, text))

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]


    # =====================================================
    # FALLBACKS SIMPLES
    # =====================================================

    def _apply_fallbacks(self, items, result):
        """
        Complète les champs manquants avec des règles simples sur le texte brut.
        """
        for item in items:
            t = item["text"].strip()

            # Nationalité : chercher "FRANCAISE" ou "FRA" standalone
            if "nationalite" not in result:
                if re.search(r"\bFRANCAISE\b|\bFRANÇAISE\b", t, re.IGNORECASE):
                    result["nationalite"] = "FRANCAISE"
                elif t.upper() == "FRA":
                    result["nationalite"] = "FRA"

        # Numéro document : préférer une chaîne alphanumérique mixte (lettres + chiffres)
        # plutôt qu'un pur entier (qui serait une date ou un autre nombre)
        if "numero_document" not in result:
            for item in items:
                t = item["text"].strip()
                if (
                    re.match(r"^[A-Z0-9]{6,15}$", t)
                    and re.search(r"[A-Z]", t)      # doit contenir des lettres
                    and re.search(r"[0-9]", t)      # doit contenir des chiffres
                    and not self._is_mrz_line(t)
                ):
                    result["numero_document"] = t
                    break

    @staticmethod
    def _is_mrz_line(text):
        """Retourne True si le texte ressemble à une ligne MRZ complète."""
        return "<" in text and len(text) >= 15


    # =====================================================
    # FUSION DE BOXES SUR LA MÊME LIGNE
    # =====================================================

    def merge_same_line_items(self, items):
        if not items:
            return items

        sorted_items = sorted(
            items,
            key=lambda item: (self.center_y(item["box"]), self.center_x(item["box"]))
        )

        merged = []
        used = set()

        for i, item in enumerate(sorted_items):

            if i in used:
                continue

            current_text = item["text"]
            current_box = item["box"]

            j = i + 1
            while j < len(sorted_items):

                if j in used:
                    j += 1
                    continue

                next_item = sorted_items[j]

                if not self.is_same_line(current_box, next_item["box"]):
                    break

                gap = next_item["box"][0][0] - current_box[1][0]

                if gap < 0 or gap > self.merge_horizontal_gap:
                    break

                candidate_text = (current_text + " " + next_item["text"]).strip()

                field_alone, _ = self.matcher.match_with_keyword(current_text)
                field_combined, _ = self.matcher.match_with_keyword(candidate_text)

                if field_combined is not None and field_alone is None:
                    current_text = candidate_text
                    current_box = self._merge_boxes(current_box, next_item["box"])
                    used.add(j)
                    j += 1
                    continue

                break

            merged.append({"text": current_text, "box": current_box})
            used.add(i)

        return merged


    @staticmethod
    def _merge_boxes(box1, box2):
        xs = [box1[0][0], box1[1][0], box2[0][0], box2[1][0]]
        ys = [box1[0][1], box1[2][1], box2[0][1], box2[2][1]]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


    # =====================================================
    # UTILITAIRES GÉOMÉTRIE
    # =====================================================

    def is_same_line(self, box1, box2):
        y1 = self.center_y(box1)
        y2 = self.center_y(box2)

        height1 = abs(box1[2][1] - box1[0][1])
        height2 = abs(box2[2][1] - box2[0][1])

        avg_height = (height1 + height2) / 2 if (height1 or height2) else 0
        tolerance = max(12, avg_height * 0.6)

        return abs(y1 - y2) <= tolerance

    def center_x(self, box):
        return (box[0][0] + box[1][0]) / 2

    def center_y(self, box):
        return (box[0][1] + box[2][1]) / 2