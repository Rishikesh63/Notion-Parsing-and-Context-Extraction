import requests
import io
import csv
import json
import PyPDF2
from PIL import Image
import pytesseract
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def ocr_image(image_url: str) -> Optional[str]:
        """Perform OCR on an image to extract text"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            # Preprocess image for better OCR results
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            text = pytesseract.image_to_string(image)
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"OCR failed for image {image_url}: {e}")
            return None
    
    @staticmethod
    def extract_file_text(file_url: str) -> Optional[str]:
        """Extract text from various file types"""
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Handle different file types
            if 'text/plain' in content_type:
                return response.text
            elif 'csv' in content_type or 'text/csv' in content_type:
                # Parse CSV
                csv_text = ""
                decoded_content = response.content.decode('utf-8')
                csv_reader = csv.reader(decoded_content.splitlines())
                for row in csv_reader:
                    csv_text += ", ".join(row) + "\n"
                return csv_text
            elif 'application/json' in content_type:
                # Parse JSON
                try:
                    json_data = response.json()
                    return json.dumps(json_data, indent=2)
                except:
                    return response.text
            # Add more file type handlers as needed
            
            return None
        except Exception as e:
            logger.error(f"Failed to extract text from file {file_url}: {e}")
            return None
    
    @staticmethod
    def extract_pdf_text(pdf_url: str) -> Optional[str]:
        """Extract text from a PDF file"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_url}: {e}")
            return None