import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

from app.pipeline import DocumentPipeline
from app.quality.image_quality_checker import ImageQualityChecker


def create_text_image(text="FACTURE N 12345", blur_kernel=None, contrast_factor=1.0):
    """
    Génère une image synthétique avec du texte pour tester l'analyseur de qualité.
    """
    img = np.full((400, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

    if contrast_factor < 1.0:
        # Réduit le contraste texte/fond
        img = np.clip(img * contrast_factor + 128 * (1 - contrast_factor), 0, 255).astype(np.uint8)

    if blur_kernel:
        img = cv2.GaussianBlur(img, blur_kernel, 0)

    return img


def test_image_quality_checker_accepts_clear_text_image(tmp_path):
    image_path = tmp_path / "clear_text.png"
    img = create_text_image("FACTURE N 12345 TOTAL 150 EUR")
    Image.fromarray(img).save(image_path)

    checker = ImageQualityChecker()
    result = checker.check(str(image_path))

    assert result["status"] == "ACCEPTED"
    assert result["score"] >= 70
    assert result["metrics"]["text_sharpness"]["score"] >= 70
    assert result["metrics"]["local_text_contrast"]["score"] >= 70
    assert result["reasons"] == []


def test_image_quality_checker_rejects_blurry_text_image(tmp_path):
    image_path = tmp_path / "blurry_text.png"
    img = create_text_image("FACTURE N 12345", blur_kernel=(31, 31))
    Image.fromarray(img).save(image_path)

    checker = ImageQualityChecker()
    result = checker.check(str(image_path))

    assert result["status"] == "REJECTED"
    assert "Netteté des caractères insuffisante (texte flou)" in result["reasons"] or result["score"] < 70


def test_image_quality_checker_rejects_low_contrast_text(tmp_path):
    image_path = tmp_path / "low_contrast.png"
    img = create_text_image("FACTURE N 12345", contrast_factor=0.05)
    Image.fromarray(img).save(image_path)

    checker = ImageQualityChecker()
    result = checker.check(str(image_path))

    assert result["status"] == "REJECTED"
    assert len(result["reasons"]) >= 1


def test_image_quality_checker_rejects_blank_image(tmp_path):
    image_path = tmp_path / "blank.png"
    img = np.full((400, 800, 3), 240, dtype=np.uint8)
    Image.fromarray(img).save(image_path)

    checker = ImageQualityChecker()
    result = checker.check(str(image_path))

    assert result["status"] == "REJECTED"
    assert any("Densité" in r or "Netteté" in r or "Contraste" in r for r in result["reasons"])


def test_pipeline_stops_when_quality_is_rejected():
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline.quality_checker = MagicMock()
    pipeline.ocr_engine = MagicMock()
    pipeline.value_extractor = MagicMock()

    pipeline.quality_checker.check.return_value = {
        "status": "REJECTED",
        "score": 35,
        "reasons": ["Netteté des caractères insuffisante (texte flou)"],
    }

    result = pipeline.process("images/test.jpeg")

    assert result["status"] == "REJECTED"
    assert result["quality"]["status"] == "REJECTED"
    assert result["data"] is None
    pipeline.ocr_engine.extract.assert_not_called()
    pipeline.value_extractor.extract.assert_not_called()
