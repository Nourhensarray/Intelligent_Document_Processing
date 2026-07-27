from app.extraction.field_matcher import match_field


class KeyValueExtractor:

    def extract(self, ocr_results):

        data = {}

        for index, item in enumerate(ocr_results):

            text = item["text"]
            field_name = match_field(text)

            if field_name is None:
                continue

            # Chercher la valeur suivante
            if index + 1 < len(ocr_results):

                next_item = ocr_results[index + 1]

                value = next_item["text"]

                # Éviter de prendre une autre clé comme valeur
                if match_field(value) is None:

                    data[field_name] = value

        return data