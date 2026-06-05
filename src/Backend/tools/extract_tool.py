import base64
import io
from typing import Literal

import pdfplumber
import pytesseract

from PIL import Image
from pdf2image import convert_from_bytes
from langchain_core.tools import tool



def ocr_confidence(image: Image.Image) -> float:
    
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    confs = []

    for conf in data["conf"]:
        try:
            conf = float(conf)
            if conf >= 0:
                confs.append(conf)
        except Exception:
            pass

    if not confs:
        return 0.0

    return round(sum(confs) / len(confs) / 100.0, 4)

def extract_image(image_bytes: bytes) -> tuple[str, float]:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image), ocr_confidence(image)


def extract_pdf(pdf_bytes: bytes) -> tuple[str, float | None]:
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
    text = "\n".join(texts).strip()
    if text:                                 
        return text, None
    
    pages = convert_from_bytes(pdf_bytes)
    all_text, confs = zip(*[(pytesseract.image_to_string(p), ocr_confidence(p)) for p in pages])
    return "\n".join(all_text), round(sum(confs) / len(confs), 4)


@tool
def extract_tool( file_type: Literal["image", "pdf"], content_b64: str) -> dict:
    """ Extract text from an image or PDF. 
        PDF falls back to OCR automatically if no native text found.
        Returns extracted_text and ocr_confidence.
        For pdf ocr_confidence is None if native text found.
        
        Args: File type (image or pdf)  and base64 encoded file content.
        
        Return: Extracted text and OCR confidence score.
    """
    file_bytes = base64.b64decode(content_b64)
    
    text, conf = extract_image(file_bytes) if file_type == "image" else extract_pdf(file_bytes)
    
    
    return {"extracted_text": text, "ocr_confidence": conf}