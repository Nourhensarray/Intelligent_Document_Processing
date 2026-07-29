import re
import unicodedata
from rapidfuzz import fuzz


class FieldMatcher:

    def __init__(self):

        self.fields = {

            "nom": [
                "nom",
                "vom",
                "nom/surname",
                "nom / surname",
                "surname",
                "family name",
                "last name",
                "name",
                "mbiemri",
                "nom d'usage",
                "nomdusage",
                "nomdusage1a"
            ],

            "prenom": [
                "prenom",
                "prénom",
                "prenoms",
                "prénoms",
                "first name",
                "given name",
                "given names",
                "emri"
            ],

            "date_naissance": [
                "date de naissance",
                "date naissance",
                "birth date",
                "date of birth",
                "dob",
                "dute depalpance",
                "date depalpance",
                "datelindja",
                "datélindia",
                "datedenaisso"
            ],

            "lieu_naissance": [
                "lieu de naissance",
                "place of birth",
                "birth place",
                "vendlindja",
                "lieudenaissance"
            ],

            "nationalite": [
                "nationalite",
                "nationalité",
                "nationality",
                "nationite",
                "shtetesia"
            ],

            "numero_document": [
                "numero document",
                "numéro document",
                "document number",
                "passport number",
                "id number",
                "n° document",
                "ndudocumento",
                "n° du document",
                "no document",
                "nr leternjoftim",
                "nr personal",
                "ndudoclment",
                "ndudoclmentloeno",
                "card no"
            ],

            "date_delivrance": [
                "date de délivrance",
                "date de delivrance",
                "date of issue",
                "issue date",
                "data e leshimit",
                "dataleshimit"
            ],

            "date_expiration": [
                "date d'expiration",
                "date expiration",
                "date of expiry",
                "expiry date",
                "expiration date",
                "dateexpir",
                "dae drplration",
                "data e skadimit"
            ],
            
            "sexe": [
                "sexe",
                "sex",
                "gjinia"
            ],

            "adresse": [
                "adresse",
                "address",
                "adresa"
            ]
        }


    def normalize(self, text):
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    def match(self, text):
        field, _ = self.match_with_keyword(text)
        return field


    def match_with_keyword(self, text):
        if not text:
            return None, None

        # S'assurer de nettoyer les étiquettes bilingues FR/EN (ex: "NOM/Smam" -> "NOM", "NATIONALITE/A" -> "NATIONALITE")
        cleaned_label = text
        if "/" in text:
            parts = text.split("/")
            if len(parts[0].strip()) >= 2:
                cleaned_label = parts[0].strip()

        normalized_text = self.normalize(cleaned_label)
        if not normalized_text:
            return None, None

        # =========================================
        # 1. MATCHING EXACT
        # =========================================

        for field, keywords in self.fields.items():
            for keyword in keywords:
                normalized_keyword = self.normalize(keyword)
                if normalized_text == normalized_keyword:
                    return field, keyword

        # =========================================
        # 2. MATCHING PAR MOTS DU MOT-CLÉ
        # =========================================

        text_words = normalized_text.split()

        for field, keywords in self.fields.items():
            for keyword in keywords:
                normalized_keyword = self.normalize(keyword)
                keyword_words = normalized_keyword.split()

                if all(word in text_words for word in keyword_words):
                    return field, keyword

        # =========================================
        # 3. FUZZY MATCHING INTELLIGENT
        # =========================================

        best_field = None
        best_keyword = None
        best_score = 0

        for field, keywords in self.fields.items():
            for keyword in keywords:
                normalized_keyword = self.normalize(keyword)
                keyword_words = normalized_keyword.split()

                if len(keyword_words) == 1:
                    keyword_score = self.best_word_score(text_words, keyword_words[0])
                    if keyword_score >= 82 and keyword_score > best_score:
                        best_score = keyword_score
                        best_field = field
                        best_keyword = keyword
                else:
                    scores = []
                    for keyword_word in keyword_words:
                        word_score = self.best_word_score(text_words, keyword_word)
                        scores.append(word_score)

                    if scores:
                        score = sum(scores) / len(scores)
                        if score >= 75 and score > best_score:
                            best_score = score
                            best_field = field
                            best_keyword = keyword

        if best_field is not None:
            return best_field, best_keyword

        return None, None


    def best_word_score(self, text_words, keyword_word):
        best_score = 0
        for text_word in text_words:
            score = fuzz.ratio(text_word, keyword_word)
            if score > best_score:
                best_score = score
        return best_score