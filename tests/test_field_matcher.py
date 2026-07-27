from app.extraction.field_matcher import FieldMatcher


matcher = FieldMatcher()


tests = [

    "Nom",

    "NOM",

    "Prenom",

    "Prenom/Cwtpinc12",

    "Date de naissance",

    "Nationality",

    "Nationite/ncbonFor",

    "Dute.depalpance",

    "Autontea",

    "Dae drplration/Dc",

    "SeSOTaerpCoueurdesm",

    "Random text"
]


for text in tests:

    result = matcher.match(text)

    print(f"{text} -> {result}")