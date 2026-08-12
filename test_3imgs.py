from app.pipeline import DocumentPipeline

p = DocumentPipeline()

for img in ["images/female_.jpg", "images/male.jpg", "images/male_11.jpg"]:
    print(f"\n{'='*60}")
    print(f"IMAGE: {img}")
    print(f"{'='*60}")
    r = p.process(img)
    print(f"Status: {r['status']}")
    print(f"Data: {r['data']}")
    ocr_zones = r.get('ocr', [])
    print(f"OCR zones: {len(ocr_zones)}")
    for i, zone in enumerate(ocr_zones):
        if isinstance(zone, dict):
            print(f"  [{i}] '{zone.get('text', '')}'")
