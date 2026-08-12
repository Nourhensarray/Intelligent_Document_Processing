import re
from app.extraction.field_matcher import FieldMatcher

class ValueExtractor:
    """
    Extracteur unifié (V2) avec support MRZ intégré et validation.
    """

    def __init__(self, matcher: "FieldMatcher | None" = None):
        # Réutilise l'instance FieldMatcher du pipeline si fournie,
        # sinon en crée une (compatibilité avec l'usage standalone).
        self.matcher = matcher if matcher is not None else FieldMatcher()
        
        # Patterns de validation par champ
        self._validators = {
            "nom":              re.compile(r"^[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ\u0600-\u06FF][a-zA-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇàâäéèêëîïôùûüç\-\s'\u0600-\u06FF]{1,40}$"),
            "prenom":           re.compile(r"^[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ\u0600-\u06FF][a-zA-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇàâäéèêëîïôùûüç,\-\s'\u0600-\u06FF]{1,40}$"),
            "nationalite":      re.compile(r"^(?!.*nationalit)(?!.*nacionalidad)(?!.*nationality)[a-zA-ZÀ-ÿ\s/\-]{3,30}$", re.IGNORECASE),
            "date_naissance":   re.compile(r"^\d{1,2}[\.\-\/\s]+\d{1,2}[\.\-\/\s]+\d{2,4}$|^\d{8}$|^\d{1,2}[\s]*[A-Za-z]{3}(?:/[A-Za-z]{3})?[\s]*\d{4}$"),
            "lieu_naissance":   re.compile(r"^[a-zA-ZÀ-ÿ\-\s\,\.\u0600-\u06FF]{2,40}$", re.IGNORECASE),
            "date_delivrance":  re.compile(r"^\d{1,2}[\.\-\/\s]+\d{1,2}[\.\-\/\s]+\d{2,4}$|^\d{8}$|^\d{1,2}[\s]*[A-Za-z]{3}(?:/[A-Za-z]{3})?[\s]*\d{4}$"),
            "date_expiration":  re.compile(r"^\d{1,2}[\.\-\/\s]+\d{1,2}[\.\-\/\s]+\d{2,4}$|^\d{8}$|^\d{1,2}[\s]*[A-Za-z]{3}(?:/[A-Za-z]{3})?[\s]*\d{4}$"),
            "numero_document":  re.compile(r"^[A-Z0-9]{6,15}$"),
            "sexe":             re.compile(r"^[MF]$", re.IGNORECASE),
            "adresse":          re.compile(r"^[a-zA-Z0-9À-ÿ\-\s\,\.\u0600-\u06FF]{5,100}$", re.IGNORECASE),
        }

    def _is_valid(self, field, value):
        if field not in self._validators:
            return True
        return bool(self._validators[field].match(value.strip()))

    def _looks_like_field_label(self, value):
        if not value:
            return False

        normalized_value = self.matcher.normalize(value)
        if not normalized_value:
            return False

        field, keyword = self.matcher.match_with_keyword(value)
        if field is None or not keyword:
            return False

        normalized_keyword = self.matcher.normalize(keyword)
        if normalized_value == normalized_keyword:
            return True

        words = normalized_value.split()
        if len(words) <= 3 and all(self.matcher.match_with_keyword(word)[0] is not None for word in words):
            return True

        return False

    def _valid_field_value(self, field, value):
        if not value:
            return False

        value = str(value).strip()
        if not value:
            return False

        if self._looks_like_field_label(value):
            return False

        return self._is_valid(field, value)

    def extract(self, layout):
        result = {}

        # --- 1. Extraction MRZ (Haute Priorité) ---
        # Aplatir les items du layout pour récupérer la MRZ
        all_items = []
        for line in layout:
            if line and "items" in line:
                all_items.extend(line["items"])
        self._extract_from_mrz(all_items, result)

        # --- 2. Extraction via Layout (Clé-Valeur Spatiale) ---
        for index, line in enumerate(layout):
            if not line:
                continue

            items = line.get("items") or []
            next_line_items = []
            if index + 1 < len(layout):
                next_line_items = layout[index + 1].get("items") or []

            if items:
                line_result = self._extract_from_items(items, next_line_items)
            else:
                line_result = self._extract_from_text(line.get("text", ""))

            # Fusionner en gardant la valeur valide la plus longue (si non existante)
            for field, value in line_result.items():
                if field and value:
                    val = str(value).strip(" :.-\u00a0/")
                    if not val or (len(val) < 2 and field != "sexe"):
                        continue
                        
                    if self._valid_field_value(field, val):
                        if field in ["date_naissance", "date_expiration", "date_delivrance"]:
                            val = self._format_date(val)
                        
                        if field not in result:
                            result[field] = val
                        else:
                            # Ne jamais écraser une valeur MRZ par une valeur lue (souvent de moins bonne qualité)
                            # Or, MRZ est appelé en premier. Donc si une valeur existe déjà, elle vient 
                            # probablement de la MRZ. Si elle est très courte (erreur de parsage ?), on peut remplacer.
                            if len(val) > len(result[field]) and field not in ["date_naissance", "date_expiration", "date_delivrance", "nom", "prenom", "numero_document", "nationalite"]:
                                result[field] = val

        return result

    def _extract_from_items(self, items, next_line_items=None):
        result = {}
        sorted_items = sorted(items, key=lambda item: self.center_x(item["box"]))

        labels = []
        for index, item in enumerate(sorted_items):
            field, keyword = self.matcher.match_with_keyword(item["text"])
            if field is not None:
                labels.append((index, field, keyword))

        if not labels:
            return self._extract_from_text(" ".join(item["text"] for item in sorted_items))

        for index, field, keyword in labels:
            if field in result:
                continue

            value = self._extract_value_from_line(sorted_items, index, labels, next_line_items)
            if value:
                result[field] = value

        return result

    def _extract_value_from_line(self, sorted_items, label_index, labels, next_line_items=None):
        label_item = sorted_items[label_index]
        field, keyword = self.matcher.match_with_keyword(label_item["text"])
        if field is None:
            return ""

        text_after_label = self._after_label(label_item["text"], keyword)
        # S'il y a un "/" dans le texte (ex: "NOM/SURNAME"), on risque de capturer "SURNAME" comme valeur.
        if text_after_label and "/" in label_item["text"]:
            parts = label_item["text"].split("/", 1)
            # Si le texte après "/" fait moins de 10 caractères ou n'a pas d'espace, c'est probablement un sous-titre
            if len(parts[1].strip()) < 15 and " " not in parts[1].strip():
                text_after_label = ""
            elif not self._is_valid(field, text_after_label):
                text_after_label = ""

        if text_after_label and self._valid_field_value(field, text_after_label):
            return text_after_label

        next_label_index = next((pos for pos, _, _ in labels if pos > label_index), len(sorted_items))
        candidate_parts = [item["text"] for item in sorted_items[label_index + 1:next_label_index]]
        candidate = " ".join(candidate_parts).strip()

        if candidate and not self._looks_like_field_label(candidate):
            if candidate and not (len(labels) > 1 and next_line_items):
                if self._valid_field_value(field, candidate):
                    return candidate

        if next_line_items:
            next_row_value = self._extract_value_from_next_row(label_item, next_line_items)
            if next_row_value and self._valid_field_value(field, next_row_value):
                return next_row_value

        combined = " ".join(item["text"] for item in sorted_items[label_index:next_label_index])
        extracted = self._after_label(combined, keyword)
        if extracted and self._valid_field_value(field, extracted):
            return extracted

        # fallback: if label is first token and rest of line are values and no next row mapping exists
        if not next_line_items and label_index == 0 and len(sorted_items) > 1:
            fb_val = " ".join(item["text"] for item in sorted_items[1:]).strip()
            if fb_val and not self._looks_like_field_label(fb_val) and self._valid_field_value(field, fb_val):
                return fb_val

        # Si rien de valide n'a été trouvé, mais qu'on avait un candidat, on le retourne pour voir
        if text_after_label and not self._looks_like_field_label(text_after_label):
            return text_after_label
        if candidate and not self._looks_like_field_label(candidate):
            return candidate
        if next_line_items:
            next_row_value = self._extract_value_from_next_row(label_item, next_line_items)
            if next_row_value and not self._looks_like_field_label(next_row_value):
                return next_row_value
        return ""

    def _extract_value_from_next_row(self, label_item, next_line_items):
        if not next_line_items:
            return ""

        label_x = self.center_x(label_item["box"])
        sorted_next_items = sorted(next_line_items, key=lambda item: self.center_x(item["box"]))

        best_item = min(
            sorted_next_items,
            key=lambda item: abs(self.center_x(item["box"]) - label_x)
        )

        return best_item["text"].strip()

    def _extract_from_text(self, text):
        result = {}
        field, keyword = self.matcher.match_with_keyword(text)
        if field is None:
            return result

        value = self._after_label(text, keyword)
        if value and self._valid_field_value(field, value):
            result[field] = value

        return result

    def _after_label(self, text, keyword):
        if not keyword:
            return text.strip()

        pattern = re.compile(r"^\s*" + re.escape(keyword) + r"\s*[:\-–—]?\s*(.*)$", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" \t\n\r\u00a0:-.")

        text = text.strip()
        if text.lower().startswith(keyword.lower()):
            return text[len(keyword):].strip(" \t\n\r\u00a0:-.")

        return ""

    @staticmethod
    def center_x(box):
        return (box[0][0] + box[1][0]) / 2

    # =====================================================
    # MRZ PARSING (From V1)
    # =====================================================

    def _extract_from_mrz(self, items, result):
        mrz_lines = []
        for item in items:
            t = item["text"].strip().upper()
            if "<" in t and len(t) >= 15:
                mrz_lines.append(t)

        if not mrz_lines:
            return

        for line in mrz_lines:
            is_name_line = False
            if re.match(r"^P[<A-Z][A-Z]{3}", line):
                is_name_line = True
            elif re.match(r"^<[A-Z]{3}", line) and "<<" in line:
                is_name_line = True
            elif re.match(r"^[A-Z]{3}[A-Z<]*<<", line):
                is_name_line = True

            if is_name_line:
                self._parse_mrz_name_line(line, result)
            elif re.match(r"^(ID[A-Z]{3}|I<[A-Z]{3})", line):
                self._parse_mrz_name_line(line, result)
            else:
                self._parse_mrz_data_line(line, result)

    def _parse_mrz_name_line(self, line, result):
        _COUNTRY_NAMES = {
            "USA": "AMÉRICAINE",    "CHE": "SUISSE",        "BRA": "BRÉSILIENNE",
            "FRA": "FRANÇAISE",    "DEU": "ALLEMANDE",     "GBR": "BRITANNIQUE",
            "ESP": "ESPAGNOLE",    "ITA": "ITALIENNE",     "PRT": "PORTUGAISE",
            "BEL": "BELGE",        "NLD": "NÉERLANDAISE",  "CAN": "CANADIENNE",
            "AUS": "AUSTRALIENNE", "MAR": "MAROCAINE",     "ALB": "ALBANAISE",
            "ZAF": "SUD-AFRICAINE","TUN": "TUNISIENNE",    "DZA": "ALGÉRIENNE",
            "EGY": "ÉGYPTIENNE",   "SEN": "SÉNÉGALAISE",   "CIV": "IVOIRIENNE",
            "CMR": "CAMEROUNAISE", "MLI": "MALIENNE",      "NER": "NIGÉRIENNE",
            "BFA": "BURKINABÈ",    "GIN": "GUINÉENNE",     "COD": "CONGOLAISE",
            "MDG": "MALGACHE",     "MUS": "MAURICIENNE",   "RWA": "RWANDAISE",
            "MEX": "MEXICAINE",    "ARG": "ARGENTINE",     "COL": "COLOMBIENNE",
            "IND": "INDIENNE",     "CHN": "CHINOISE",      "JPN": "JAPONAISE",
            "RUS": "RUSSE",        "TUR": "TURQUE",        "PAK": "PAKISTANAISE",
        }

        country_code = None
        clean_line = line
        
        m_standard = re.match(r"^P[<A-Z]([A-Z]{3})", line)
        m_no_p = re.match(r"^<([A-Z]{3})", line)
        m_no_prefix = re.match(r"^([A-Z]{3})", line)
        
        if m_standard:
            country_code = m_standard.group(1)
            clean_line = line[5:]
        elif m_no_p and "<<" in line:
            country_code = m_no_p.group(1)
            clean_line = line[4:]
        elif m_no_prefix and "<<" in line:
            country_code = m_no_prefix.group(1)
            clean_line = line[3:]
            
        if country_code and "nationalite" not in result:
            result["nationalite"] = _COUNTRY_NAMES.get(country_code, country_code)

        parts = clean_line.split("<<", 1)

        if parts[0] and "nom" not in result:
            surname = parts[0].replace("<", " ").strip()
            if surname and re.match(r"^[A-Z\- ]{2,30}$", surname):
                result["nom"] = surname

        if len(parts) >= 2 and parts[1] and "prenom" not in result:
            given = re.sub(r"<+", " ", parts[1]).strip()
            if given:
                result["prenom"] = given.title()

    def _parse_mrz_data_line(self, line, result):
        clean = line.replace(" ", "")
        length = len(clean)

        # ---- TD3 (passeport) : 44 chars ----
        if length >= 38:
            td3_match = re.search(r"([A-Z0-9]{6,9})<?(?:\d|<<)?([A-Z]{3})(\d{6})\d?([MF<])(\d{6})", clean)
            if td3_match:
                doc = td3_match.group(1)
                if doc and "numero_document" not in result:
                    result["numero_document"] = doc
                
                country_code = td3_match.group(2)
                _COUNTRY_NAMES = {
                    "USA": "AMÉRICAINE",    "CHE": "SUISSE",        "BRA": "BRÉSILIENNE",
                    "FRA": "FRANÇAISE",    "DEU": "ALLEMANDE",     "GBR": "BRITANNIQUE",
                }
                if country_code and "nationalite" not in result:
                    result["nationalite"] = _COUNTRY_NAMES.get(country_code, country_code)
                    
                dob_str = td3_match.group(3)
                if dob_str and "date_naissance" not in result:
                    result["date_naissance"] = self._format_mrz_date(dob_str, is_birth=True)
                    
                sex_marker = td3_match.group(4)
                if sex_marker in ("M", "F") and "sexe" not in result:
                    result["sexe"] = sex_marker
                    
                exp_str = td3_match.group(5)
                if exp_str and "date_expiration" not in result:
                    result["date_expiration"] = self._format_mrz_date(exp_str, is_birth=False)
                return

            doc = clean[:9].rstrip("<")
            if doc and re.match(r"^[A-Z0-9]{3,9}$", doc) and "numero_document" not in result:
                result["numero_document"] = doc

            sex_marker = clean[20:21] if length > 20 else ""
            if sex_marker in ("M", "F"):
                if "sexe" not in result:
                    result["sexe"] = sex_marker
                dob_str = clean[13:19]
                exp_str = clean[21:27]
                if re.match(r"^\d{6}$", dob_str) and "date_naissance" not in result:
                    result["date_naissance"] = self._format_mrz_date(dob_str, is_birth=True)
                if re.match(r"^\d{6}$", exp_str) and "date_expiration" not in result:
                    result["date_expiration"] = self._format_mrz_date(exp_str, is_birth=False)
            return

        # ---- TD1 (CNI) : ~30 chars ----
        if length >= 15:
            match_doc = re.match(r"^([A-Z0-9]{6,9})", clean)
            if match_doc and "numero_document" not in result:
                doc = match_doc.group(1).rstrip("<")
                if len(doc) >= 6:
                    result["numero_document"] = doc

            td1_match = re.search(r"(\d{6})\d?([MF<])(\d{6})", clean)
            if td1_match:
                if "date_naissance" not in result:
                    result["date_naissance"] = self._format_mrz_date(td1_match.group(1), is_birth=True)
                if "date_expiration" not in result:
                    result["date_expiration"] = self._format_mrz_date(td1_match.group(3), is_birth=False)
                if "sexe" not in result and td1_match.group(2) in ("M", "F"):
                    result["sexe"] = td1_match.group(2)

    @staticmethod
    def _format_mrz_date(d, is_birth=True):
        if len(d) != 6:
            return d
        yy, mm, dd = d[:2], d[2:4], d[4:]
        if is_birth:
            year = f"19{yy}" if int(yy) > 26 else f"20{yy}"
        else:
            year = f"20{yy}"
        return f"{dd}/{mm}/{year}"

    @staticmethod
    def _format_date(val):
        _MONTHS = {
            "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
            "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
            "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
        }
        # Clean up spaces and duplicate months (e.g., "05 NOV/NOV 2023" -> "05NOV2023")
        clean_val = re.sub(r"\s+", "", val.strip().upper())
        # Handle duplicated month separated by slash e.g. "NOV/NOV"
        clean_val = re.sub(r"([A-Z]{3})/\1", r"\1", clean_val)
        
        m = re.match(r"^(\d{1,2})([A-Z]{3})(\d{4})$", clean_val)
        if m:
            month = _MONTHS.get(m.group(2))
            if month:
                return f"{m.group(1).zfill(2)}/{month}/{m.group(3)}"
        clean_digit = re.sub(r"[^\d]", "", val)
        if len(clean_digit) == 8:
            return f"{clean_digit[:2]}/{clean_digit[2:4]}/{clean_digit[4:]}"
        return val
