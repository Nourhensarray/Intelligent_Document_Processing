import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.extraction.value_extractor import ValueExtractor
from app.extraction.value_extractor_v2 import ValueExtractorV2
from app.layout.document_layout import DocumentLayoutBuilder

ocr_data = [
    {
        'text': 'Nom',
        'confidence': 0.99,
        'box': [[100, 100], [150, 100], [150, 120], [100, 120]]
    },
    {
        'text': 'HUMBERT',
        'confidence': 0.99,
        'box': [[100, 140], [180, 140], [180, 160], [100, 160]]
    },
    {
        'text': 'Prenom',
        'confidence': 0.99,
        'box': [[300, 100], [370, 100], [370, 120], [300, 120]]
    },
    {
        'text': 'FLAVIEN',
        'confidence': 0.99,
        'box': [[300, 140], [380, 140], [380, 160], [300, 160]]
    },
    {
        'text': 'Nationality',
        'confidence': 0.99,
        'box': [[500, 100], [600, 100], [600, 120], [500, 120]]
    },
    {
        'text': 'FRANCAISE',
        'confidence': 0.99,
        'box': [[500, 140], [600, 140], [600, 160], [500, 160]]
    }
]

print('ValueExtractor result:')
print(ValueExtractor().extract(ocr_data))

layout = DocumentLayoutBuilder().build(ocr_data)
print('Layout:')
print(layout)

print('ValueExtractorV2 result:')
print(ValueExtractorV2().extract(layout))
