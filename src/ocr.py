import os
import json
import logging
from typing import Dict, List, Any

import numpy as np
import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import pytesseract
from PIL import Image

# Get logger for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Tesseract command path if needed
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD", pytesseract.pytesseract.tesseract_cmd
)


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the conversion
        
    Returns:
        List of PIL Image objects, one per page
    """
    try:
        return convert_from_path(pdf_path, dpi=dpi)
    except PDFInfoNotInstalledError:
        logger.error(
            "Poppler is not installed or not found in PATH. "
            "Please install Poppler and ensure it's in your PATH environment variable."
        )
        return []
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return []


def extract_text_blocks_ocr(image: Image.Image) -> List[str]:
    """
    Extract text blocks from an image using OCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of text blocks
    """
    try:
        # Run OCR to get detailed text data
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Group words by line number to form text blocks
        line_texts = {}
        for i, line_num in enumerate(ocr_data['line_num']):
            if line_num >= 0:  # Skip entries with invalid line numbers
                # Filter out low-confidence words and empty strings
                if ocr_data['conf'][i] > 30 and ocr_data['text'][i].strip():
                    if line_num not in line_texts:
                        line_texts[line_num] = []
                    line_texts[line_num].append(ocr_data['text'][i])
        
        # Join words in each line to form text blocks
        text_blocks = []
        for line_num in sorted(line_texts.keys()):
            line_text = ' '.join(line_texts[line_num]).strip()
            if line_text:  # Only add non-empty text blocks
                text_blocks.append(line_text)
        
        return text_blocks
    except Exception as e:
        logger.error(f"Error extracting text with OCR: {e}")
        return []


def extract_tables_ocr(image: Image.Image) -> List[List[List[str]]]:
    """
    Extract tables from an image using OpenCV for line detection and OCR for text extraction.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of tables, where each table is a list of rows, and each row is a list of cells
    """
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Get image dimensions
        height, width = thresh.shape
        
        # Define kernel sizes for line detection
        kernel_len_v = np.array(gray).shape[1] // 100
        kernel_len_h = np.array(gray).shape[0] // 100
        
        # Define vertical and horizontal kernels
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))
        
        # Detect vertical lines
        vertical_lines = cv2.erode(thresh, kernel_v, iterations=3)
        vertical_lines = cv2.dilate(vertical_lines, kernel_v, iterations=3)
        
        # Detect horizontal lines
        horizontal_lines = cv2.erode(thresh, kernel_h, iterations=3)
        horizontal_lines = cv2.dilate(horizontal_lines, kernel_h, iterations=3)
        
        # Combine horizontal and vertical lines
        combined = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (keep only those that might be table cells)
        min_area = width * height * 0.0005  # Minimum area threshold for a cell
        max_area = width * height * 0.5     # Maximum area threshold for a cell
        min_width = 10  # Minimum width in pixels
        min_height = 10  # Minimum height in pixels
        
        cell_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if (min_area < area < max_area) and (w >= min_width) and (h >= min_height):
                cell_contours.append(cnt)
        
        # If not enough cells found, assume no table
        if len(cell_contours) < 4:  # At least a 2x2 table
            return []
        
        # Get bounding rectangles for cells
        cell_boxes = [cv2.boundingRect(cnt) for cnt in cell_contours]
        
        # Define row threshold (cells within this y-distance belong to the same row)
        row_threshold = height * 0.02
        
        # Group cells by row
        rows = []
        current_row = []
        cell_boxes.sort(key=lambda box: box[1])  # Sort by y-coordinate
        
        current_y = cell_boxes[0][1]
        for box in cell_boxes:
            x, y, w, h = box
            if abs(y - current_y) > row_threshold:
                # New row
                if current_row:
                    current_row.sort(key=lambda cell: cell[0])  # Sort by x-coordinate
                    rows.append(current_row)
                current_row = [box]
                current_y = y
            else:
                current_row.append(box)
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda cell: cell[0])
            rows.append(current_row)
        
        # Extract text from each cell
        table = []
        for row_boxes in rows:
            row_text = []
            for x, y, w, h in row_boxes:
                # Extract cell region from original image
                cell_img = image.crop((x, y, x+w, y+h))
                
                # Apply OCR to cell
                cell_text = pytesseract.image_to_string(cell_img).strip()
                row_text.append(cell_text)
                
                # Release memory
                cell_img.close()
            
            if row_text:  # Only add non-empty rows
                table.append(row_text)
        
        return [table] if table else []
    
    except Exception as e:
        logger.error(f"Error extracting tables with OCR: {e}")
        return []


def parse_pdf_ocr(pdf_path: str, dpi: int = 300) -> Dict[str, Any]:
    """
    Parse a PDF file using OCR and extract text blocks and tables.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the PDF to image conversion
        
    Returns:
        A dictionary with the same structure as parse_pdf:
        {
            "pages": [
                {
                    "page_num": 1,
                    "text_blocks": ["First line of text", "Next paragraph…", …],
                    "tables": [
                        [["Header1","Header2",…], ["row1col1","row1col2",…], …],
                        …  # one entry per table found on this page
                    ]
                },
                ...
            ]
        }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    result = {"pages": []}
    
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path, dpi=dpi)
    
    # Process each page
    for i, image in enumerate(images):
        page_num = i + 1
        
        # Extract text and tables
        text_blocks = extract_text_blocks_ocr(image)
        tables = extract_tables_ocr(image)
        
        # Add page data to result
        page_data = {
            "page_num": page_num,
            "text_blocks": text_blocks,
            "tables": tables
        }
        result["pages"].append(page_data)
        
        # Release memory
        image.close()
    
    return result


def save_parsed_pdf_ocr(pdf_path: str, output_path: str, dpi: int = 300) -> None:
    """Parse a PDF using OCR and save the result to a JSON file."""
    try:
        parsed_data = parse_pdf_ocr(pdf_path, dpi=dpi)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(parsed_data, file, ensure_ascii=False, indent=2)
        
        logger.info(f"OCR parsed PDF saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to parse PDF with OCR: {e}", exc_info=True)


if __name__ == "__main__":
    import sys
    
    # Configure logging only when running as script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <pdf_path> [output_path] [dpi]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_pdf_ocr.json"
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    try:
        save_parsed_pdf_ocr(pdf_path, output_path, dpi)
    except Exception as e:
        logger.error(f"Unhandled exception in OCR processing: {e}", exc_info=True)
        sys.exit(1)