import re
from rapidfuzz import process, fuzz


class OCRPostProcessor:

    def __init__(self):

        self.dictionary = [

            # Nom
            "nom",
            "surname",
            "family name",
            "last name",
            "familienname",
            "sobrenome",

            # Prénom
            "prenom",
            "prénom",
            "given name",
            "given names",
            "first name",
            "forenames",
            "vorname",

            # Nationalité
            "nationalite",
            "nationalité",
            "nationality",
            "citizenship",

            # Sexe
            "sexe",
            "sex",
            "gender",

            # Date naissance
            "date de naissance",
            "date of birth",
            "birth date",
            "dob",

            # Date expiration
            "date expiration",
            "date d'expiration",
            "date of expiry",
            "expiry date",

            # Numéro document
            "document number",
            "passport number",
            "numero document",
            "identity number",
        ]

    def process(self, ocr_data):

        cleaned = []

        for item in ocr_data:

            text = item["text"]

            text = self.fix_common_errors(text)

            text = self.correct_keywords(text)

            item["text"] = text

            cleaned.append(item)

        return cleaned

    def fix_common_errors(self, text):

        replacements = {

            "|": "I",
            "§": "S",
            "€": "C",
            "«": "<",
            "»": ">",
            "—": "-",
            "_": "-",

        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def correct_keywords(self, text):

        candidate, score, _ = process.extractOne(
            text,
            self.dictionary,
            scorer=fuzz.ratio
        )

        if score > 88:
            return candidate

        return text