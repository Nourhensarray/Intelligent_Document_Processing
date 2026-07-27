from app.ocr.paddle_engine import PaddleEngine


ocr = PaddleEngine()

result = ocr.extract("images/test.jpeg")

for item in result:
    print("\n--------------------")
    print("Texte :", item["text"])
    print("Confiance :", item["confidence"])
    print("Position :", item["box"])