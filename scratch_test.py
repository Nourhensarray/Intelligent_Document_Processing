import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.extraction.field_matcher import FieldMatcher
from app.extraction.value_extractor import ValueExtractor

matcher = FieldMatcher()
extractor = ValueExtractor()

print("Testing DATEEXPIR./ExiY.")
field, kw = matcher.match_with_keyword("DATEEXPIR./ExiY.")
print(f"Matched field: {field}, Keyword: {kw}")

print("Testing NDUDOCLMENTLOENO")
field, kw = matcher.match_with_keyword("NDUDOCLMENTLOENO")
print(f"Matched field: {field}, Keyword: {kw}")

# Also let's check fuzzy matching score
from rapidfuzz import fuzz
print(f"Fuzz NDUDOCLMENTLOENO vs ndudocumento: {fuzz.ratio('ndudoclmentloeno', 'ndudocumento')}")
print(f"Fuzz DATEEXPIR vs dateoexpir: {fuzz.ratio('dateexpir', 'dateoexpir')}")
