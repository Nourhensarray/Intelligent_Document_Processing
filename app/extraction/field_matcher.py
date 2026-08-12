import difflib
import re
import unicodedata
from functools import lru_cache


# Cache global des normalisations de texte (évite de recalculer pour les
# mêmes tokens OCR qui reviennent constamment sur chaque ligne du layout).
@lru_cache(maxsize=2048)
def _normalize_cached(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class FieldMatcher:

    def __init__(self):
        # Cache des résultats de match_with_keyword() pour éviter de refaire
        # le fuzzy matching sur les mêmes tokens OCR (labels répétés sur chaque ligne).
        self._match_cache: dict = {}

        self.fields = {

            "nom": [
                # Français (CNI/passeport FR)
                "nom",
                "vom",
                "nom/surname",
                "nom / surname",
                "nom d'usage",
                "nomdusage",
                "nomdusage1a",
                # Anglais (USA, UK, passeports anglophones)
                "surname",
                "surmame",  # OCR variant ZAF
                "sumame",   # OCR variant ZAF
                "family name",
                "last name",
                "name",
                # Allemand / passeport suisse (CHE)
                "familienname",
                "nachname",
                # Portugais / passeport brésilien (BRA)
                "sobrenome",
                "nome",
                "apelido",
                # Italien
                "cognome",
                # Albanais
                "mbiemri",
                # Slovaque
                "priezvisko",
                # Afrique du Sud (ZAF)
                "surname/nom",
                "surmame/nom",
                # Arabe (CIN Tunisienne)
                "اللقب",
            ],

            "prenom": [
                # Français
                "prenom",
                "prénom",
                "prenoms",
                "prénoms",
                # Anglais (USA)
                "first name",
                "given name",
                "given names",
                "given name(s)",
                "first and middle names",
                # Allemand / passeport suisse (CHE)
                "vorname",
                "vornamen",
                # Portugais / passeport brésilien (BRA)
                "nome do titular",
                "primeiro nome",
                # Albanais
                "emri",
                # Slovaque
                "meno",
                # Afrique du Sud (ZAF)
                "given name",
                "forenames",
                "voornamen",
                # Arabe (CIN Tunisienne)
                "الاسم",
            ],

            "date_naissance": [
                # Français
                "date de naissance",
                "date naissance",
                "datedenaisso",
                # Anglais (USA)
                "birth date",
                "date of birth",
                "date.of birth",
                "date ofbirth",
                "dob",
                # Allemand / passeport suisse (CHE)
                "geburtsdatum",
                "geboren",
                # Portugais / passeport brésilien (BRA)
                "data de nascimento",
                "nascimento",
                "data nasc",
                # Variantes OCR
                "dute depalpance",
                "date depalpance",
                "datelindja",
                "datélindia",
                # Slovaque
                "datum narodenia",
                # Afrique du Sud (ZAF) – labels bilingues et variantes OCR
                "date of birth/date de naissance",
                "date.ofbirth/date de naissance",
                "dateofbirth",
                "date.ofbirth",
                # Arabe (CIN Tunisienne)
                "تاريخ الولادة",
            ],

            "lieu_naissance": [
                # Français
                "lieu de naissance",
                "lieudenaissance",
                # Anglais (USA)
                "place of birth",
                "birth place",
                "city of birth",
                # Allemand / passeport suisse (CHE)
                "geburtsort",
                # Portugais / passeport brésilien (BRA)
                "local de nascimento",
                "naturalidade",
                # Albanais
                "vendlindja",
                # Afrique du Sud (ZAF)
                "place of birth/lieu de naissance",
                "place.of birth",
                # Arabe (CIN Tunisienne)
                "مكانها",
                "مكان الولادة",
            ],

            "nationalite": [
                # Français
                "nationalite",
                "nationalité",
                "nationite",
                # Anglais (USA)
                "nationality",
                "citizenship",
                # Allemand / passeport suisse (CHE)
                "staatsangehörigkeit",
                "nationalität",
                # Portugais / passeport brésilien (BRA)
                "nacionalidade",
                # Albanais
                "shtetesia",
                # Slovaque
                "statne obcianstvo",
                # Afrique du Sud (ZAF)
                "nationality/national",
                "nationality/nationalite",
            ],

            "numero_document": [
                # Français
                "numero document",
                "numéro document",
                "n° document",
                "n° du document",
                "no document",
                "ndudocumento",
                "ndudoclment",
                "ndudoclmentloeno",
                # Anglais (USA)
                "document number",
                "passport number",
                "passport no",
                "id number",
                "card no",
                "book number",
                "pass/passport",
                "pass passport",
                # Allemand / passeport suisse (CHE)
                "reisepassnummer",
                "ausweisnummer",
                "dokumentennummer",
                "pass-nr",
                # Portugais / passeport brésilien (BRA)
                "numero do passaporte",
                "número do passaporte",
                "no do passaporte",
                "rne",
                # Albanais
                "nr leternjoftim",
                "nr personal",
                # Slovaque
                "cislo",
                # Afrique du Sud (ZAF)
                "passport/passeport",
                "pass/passeport",
                "identity no",
                "identity number",
                "id no",
                "identityno",
                "dentityno",  # OCR variant
                "dentity no",
                # Arabe (CIN Tunisienne)
                "رقم بطاقة التعريف",
                "رقم بطاقة التعريف الوطنية",
                "بطاقة تعريف",
            ],

            "date_delivrance": [
                # Français
                "date de délivrance",
                "date de delivrance",
                # Anglais (USA)
                "date of issue",
                "issue date",
                "date issued",
                "issued",
                # Allemand / passeport suisse (CHE)
                "ausstellungsdatum",
                "ausgestellt am",
                # Portugais / passeport brésilien (BRA)
                "data de emissao",
                "data de emissão",
                "emissao",
                "data de expedicao",
                "datadeexpedicao",
                # Albanais
                "data e leshimit",
                "dataleshimit",
                # Slovaque
                "datum vydania",
                # Afrique du Sud (ZAF)
                "date of issue/date de delivrance",
                "date.of issue",
                "dateofissue",
                "daleossue",      # OCR variant ZAF
                "daleossue/date de devranc",
                "date de devranc",
                # Arabe (CIN Tunisienne)
                "تاريخ الاصدار",
                "تاريخ الإصدار",
            ],

            "date_expiration": [
                # Français
                "date d'expiration",
                "date expiration",
                "dateexpir",
                # Anglais (USA)
                "date of expiry",
                "expiry date",
                "expiration date",
                "date of expiration",
                "date of expiry/date d'expiration",
                # Allemand / passeport suisse (CHE)
                "ablaufdatum",
                "gültig bis",
                "gultig bis",
                # Portugais / passeport brésilien (BRA)
                "data de validade",
                "validade",
                "data validade",
                # Variantes OCR
                "dae drplration",
                "data e skadimit",
                # Slovaque
                "datum platnosti",
                # Afrique du Sud (ZAF)
                "date of expiry/date d expire",
                "deteof expiry",
                "dateof expiry",
                "date of expiry/dated expire",
                "deteof expiry/dated'expire",
            ],
            
            "sexe": [
                # Français
                "sexe",
                # Anglais (USA)
                "sex",
                "gender",
                # Allemand / passeport suisse (CHE)
                "geschlecht",
                # Portugais / passeport brésilien (BRA)
                "sexo",
                # Slovaque
                "pohlavie",
                # Albanais
                "gjinia",
                # Arabe (CIN Tunisienne)
                "الجنس",
            ],

            "adresse": [
                # Français
                "adresse",
                # Anglais (USA)
                "address",
                "permanent address",
                "home address",
                "residence",
                # Allemand / passeport suisse (CHE)
                "adresse",
                "wohnort",
                # Portugais / passeport brésilien (BRA)
                "endereco",
                "endereço",
                # Albanais
                "adresa",
                # Arabe (CIN Tunisienne)
                "العنوان",
            ]
        }


    def normalize(self, text):
        # Déléguer au cache global pour éviter les recomputations répétées.
        return _normalize_cached(text)


    def match(self, text):
        field, _ = self.match_with_keyword(text)
        return field


    def match_with_keyword(self, text):
        if not text:
            return None, None

        # ── Cache : évite le fuzzy matching répété sur les mêmes tokens OCR ──
        cached = self._match_cache.get(text)
        if cached is not None:
            return cached

        # Remplacer les points comme séparateurs dans les labels composés (Date.ofbirth -> Date ofbirth)
        text_clean = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', ' ', text)

        labels_to_test = [text_clean]
        # Nettoyer les étiquettes bilingues séparées par "/"
        # Ex: "NOM/Surname" -> On testera "NOM/Surname", "NOM" et "Surname"
        if "/" in text_clean:
            parts = text_clean.split("/")
            labels_to_test.extend([p.strip() for p in parts if len(p.strip()) >= 2])

        best_field_global = None
        best_keyword_global = None
        best_score_global = 0

        for label in labels_to_test:
            normalized_text = self.normalize(label)
            if not normalized_text:
                continue

            # =========================================
            # 1. MATCHING EXACT
            # =========================================
            for field, keywords in self.fields.items():
                for keyword in keywords:
                    normalized_keyword = self.normalize(keyword)
                    if normalized_text == normalized_keyword:
                        result = (field, keyword)
                        self._match_cache[text] = result
                        return result

            # =========================================
            # 2. MATCHING PAR MOTS DU MOT-CLÉ
            # =========================================
            text_words = normalized_text.split()
            for field, keywords in self.fields.items():
                for keyword in keywords:
                    normalized_keyword = self.normalize(keyword)
                    keyword_words = normalized_keyword.split()

                    if all(word in text_words for word in keyword_words):
                        result = (field, keyword)
                        self._match_cache[text] = result
                        return result

            # =========================================
            # 3. FUZZY MATCHING INTELLIGENT
            # =========================================
            for field, keywords in self.fields.items():
                for keyword in keywords:
                    normalized_keyword = self.normalize(keyword)
                    keyword_words = normalized_keyword.split()

                    if len(keyword_words) == 1:
                        keyword_score = self.best_word_score(text_words, keyword_words[0])
                        if keyword_score >= 82 and keyword_score > best_score_global:
                            best_score_global = keyword_score
                            best_field_global = field
                            best_keyword_global = keyword
                    else:
                        scores = []
                        for keyword_word in keyword_words:
                            word_score = self.best_word_score(text_words, keyword_word)
                            scores.append(word_score)

                        if scores:
                            score = sum(scores) / len(scores)
                            if score >= 75 and score > best_score_global:
                                best_score_global = score
                                best_field_global = field
                                best_keyword_global = keyword

        result = (best_field_global, best_keyword_global) if best_field_global is not None else (None, None)
        self._match_cache[text] = result
        return result

    def best_word_score(self, text_words, keyword_word):
        if not text_words:
            return 0
        best_score = 0
        for text_word in text_words:
            score = difflib.SequenceMatcher(None, text_word, keyword_word).ratio() * 100
            if score > best_score:
                best_score = score
        return best_score