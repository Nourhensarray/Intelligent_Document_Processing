import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.extraction.value_extractor import ValueExtractor


def test_value_extractor_rejects_label_only():
    extractor = ValueExtractor()
    layout = [
        {
            "text": "Nom",
            "items": [
                {"text": "Nom", "box": [[100, 100], [150, 100], [150, 120], [100, 120]]}
            ]
        },
        {
            "text": "Prenom",
            "items": [
                {"text": "Prenom", "box": [[100, 130], [160, 130], [160, 150], [100, 150]]}
            ]
        }
    ]

    assert extractor.extract(layout) == {}


def test_value_extractor_extracts_valid_values_after_labels():
    extractor = ValueExtractor()
    layout = [
        {
            "text": "Nom HUMBERT",
            "items": [
                {"text": "Nom", "box": [[100, 100], [150, 100], [150, 120], [100, 120]]},
                {"text": "HUMBERT", "box": [[160, 100], [240, 100], [240, 120], [160, 120]]}
            ]
        },
        {
            "text": "Prenom FLAVIEN",
            "items": [
                {"text": "Prenom", "box": [[100, 130], [170, 130], [170, 150], [100, 150]]},
                {"text": "FLAVIEN", "box": [[180, 130], [240, 130], [240, 150], [180, 150]]}
            ]
        },
        {
            "text": "Nationality FRANCAISE",
            "items": [
                {"text": "Nationality", "box": [[100, 160], [200, 160], [200, 180], [100, 180]]},
                {"text": "FRANCAISE", "box": [[210, 160], [310, 160], [310, 180], [210, 180]]}
            ]
        }
    ]

    result = extractor.extract(layout)
    assert result["nom"] == "HUMBERT"
    assert result["prenom"] == "FLAVIEN"
    assert result["nationalite"] == "FRANCAISE"
