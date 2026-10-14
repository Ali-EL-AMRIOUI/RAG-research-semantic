"""
Service OCR — wrapper autour de pytesseract.

Extrait de document_loader.py, qui mêlait extraction OCR et logique de
chargement/chunking de documents. Isolé pour tester l'OCR indépendamment
et changer de moteur (EasyOCR, PaddleOCR...) sans toucher au reste du
loader.

document_loader.py utilise maintenant CETTE classe au lieu d'appeler
pytesseract directement.
"""
import logging
from pathlib import Path
from typing import Union
from io import BytesIO

from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class OCRService:

    def __init__(self, languages: str = "fra+eng"):
        self.languages = languages

    def extract_text(self, image: Union[str, Path, bytes, BytesIO]) -> str:
        try:
            if isinstance(image, bytes):
                img = Image.open(BytesIO(image))
            elif isinstance(image, BytesIO):
                img = Image.open(image)
            else:
                img = Image.open(image)

            text = pytesseract.image_to_string(img, lang=self.languages)
            logger.info(f"OCR: {len(text)} caractères extraits")
            return text

        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return ""

    def is_available(self) -> bool:
        """
        Vérifie que le binaire tesseract est réellement installé et
        accessible (pas juste que le package Python pytesseract est
        importable — c'est un wrapper autour d'un binaire externe qui peut
        très bien manquer sur la machine).
        """
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract non disponible: {e}")
            return False