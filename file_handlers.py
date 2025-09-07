import requests
import io
import csv
import json
import pypdf  
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from typing import Optional
import logging
import os
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, base_url: str = ""):
        """
        Initialize FileHandler with optional base URL for relative paths
        
        Args:
            base_url: Base URL to resolve relative URLs against
        """
        self.base_url = base_url.rstrip('/')  
        
        # Windows-specific Tesseract configuration
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                logger.warning("Tesseract not found at default Windows path")

    def _resolve_url(self, url: str) -> str:
        """
        Resolve relative URLs against the base URL
        
        Args:
            url: URL to resolve (can be absolute or relative)
            
        Returns:
            str: Absolute URL
        """
        if url.startswith(('http://', 'https://')):
            return url  # Already absolute
        
        if url.startswith('/'):
            # Relative URL starting with /
            if not self.base_url:
                raise ValueError(f"Relative URL {url} provided but no base_url set")
            return f"{self.base_url}{url}"
        else:
            # Relative URL without leading slash
            if not self.base_url:
                raise ValueError(f"Relative URL {url} provided but no base_url set")
            return urljoin(self.base_url, url)

    def ocr_image(self, image_url: str, save_debug: bool = True, lang: str = 'eng') -> Optional[str]:
        """Perform OCR on an image to extract text, saving preprocessed images to images/ folder."""
        try:
            # Resolve relative URL
            full_url = self._resolve_url(image_url)
            logger.info(f"Downloading image from: {full_url}")

            response = requests.get(full_url, timeout=30)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Resize image for better OCR (if small)
            min_size = 800
            if image.width < min_size or image.height < min_size:
                scale = max(min_size / image.width, min_size / image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)

            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.5)

            # Sharpen image
            image = image.filter(ImageFilter.SHARPEN)

            # Binarize image (thresholding)
            threshold = 160
            image = image.point(lambda p: 255 if p > threshold else 0)

            # Save preprocessed image to images/ folder
            os.makedirs("images", exist_ok=True)
            debug_path = os.path.join("images", f"ocr_debug_{os.path.basename(image_url).split('?')[0]}")
            try:
                image.save(debug_path)
                logger.info(f"Saved preprocessed image for debugging: {debug_path}")
            except Exception as e:
                logger.warning(f"Failed to save debug image: {e}")

            # Try multiple OCR configurations for better results
            configs = [
                f'-l {lang} --oem 3 --psm 6',  # Uniform block of text
                f'-l {lang} --oem 3 --psm 11', # Sparse text
                f'-l {lang} --oem 3 --psm 8',  # Single word
            ]

            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    logger.info(f"OCR result with config '{config}': {text.strip()}")
                    if text.strip() and len(text.strip()) > len(best_text):
                        best_text = text.strip()
                except Exception as ocr_err:
                    logger.warning(f"Tesseract OCR failed with config '{config}': {ocr_err}")

            if not best_text:
                logger.warning(f"No text extracted from image: {image_url}")
            return best_text if best_text else None

        except Exception as e:
            logger.error(f"OCR failed for image {image_url}: {e}")
            return None
    
    def extract_file_text(self, file_url: str) -> Optional[str]:
        """Extract text from various file types and save files to files/ folder."""
        try:
            # Resolve relative URL
            full_url = self._resolve_url(file_url)
            logger.info(f"Downloading file from: {full_url}")

            response = requests.get(full_url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            file_extension = file_url.lower().split('.')[-1] if '.' in file_url else ''

            # Save file to files/ folder
            os.makedirs("files", exist_ok=True)
            file_name = os.path.basename(file_url).split('?')[0]
            file_path = os.path.join("files", file_name)
            try:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved file to: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to save file: {e}")

            if 'text/plain' in content_type or file_extension in ['txt', 'text']:
                return response.text
            elif 'csv' in content_type or file_extension == 'csv':
                csv_text = ""
                decoded_content = response.content.decode('utf-8')
                csv_reader = csv.reader(decoded_content.splitlines())
                for row in csv_reader:
                    csv_text += ", ".join(row) + "\n"
                return csv_text
            elif 'application/json' in content_type or file_extension == 'json':
                try:
                    json_data = response.json()
                    return json.dumps(json_data, indent=2)
                except Exception:
                    return response.text
            elif 'application/pdf' in content_type or file_extension == 'pdf':
                return self.extract_pdf_text(file_url)
            elif any(ext in content_type for ext in ['image/jpeg', 'image/png', 'image/gif', 'image/tiff', 'image/bmp']):
                return self.ocr_image(file_url)

            logger.warning(f"Unsupported file type: {content_type} for URL: {file_url}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract text from file {file_url}: {e}")
            return None
    
    def extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Extract text from a PDF file and save PDF to pdfs/ folder."""
        try:
            # Resolve relative URL
            full_url = self._resolve_url(pdf_url)
            logger.info(f"Downloading PDF from: {full_url}")

            response = requests.get(full_url, timeout=30)
            response.raise_for_status()

            # Save PDF to pdfs/ folder
            os.makedirs("pdfs", exist_ok=True)
            pdf_name = os.path.basename(pdf_url).split('?')[0]
            pdf_path = os.path.join("pdfs", pdf_name)
            try:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Saved PDF to: {pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to save PDF: {e}")

            pdf_file = io.BytesIO(response.content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            return text.strip() if text else None

        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_url}: {e}")
            return None

    def process_content_item(self, content_item: dict) -> Optional[str]:
        """
        Convenience method to process a content item directly
        
        Args:
            content_item: Dictionary with content item data
            
        Returns:
            Extracted text or None
        """
        if not isinstance(content_item, dict):
            return None
            
        item_type = content_item.get('type')
        metadata = content_item.get('metadata', {})
        url = metadata.get('url')
        
        if not url:
            return None
            
        if item_type == 'image':
            return self.ocr_image(url)
        elif item_type == 'file':
            return self.extract_file_text(url)
        elif item_type == 'pdf':
            return self.extract_pdf_text(url)
        else:
            # Try to auto-detect based on URL
            return self.extract_file_text(url)