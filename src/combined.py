import os
import json
import logging
from typing import Dict, Any

import logging
logger = logging.getLogger(__name__)

from src.parser import parse_pdf as parse_pdf_digital
from src.ocr import parse_pdf_ocr

def parse_pdf_combined(pdf_path: str, ocr_dpi: int = 300) -> Dict[str, Any]:   
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # First try digital extraction for the entire PDF
    digital_result = parse_pdf_digital(pdf_path)
    
    # Check if any pages have no content and require OCR
    ocr_needed = False
    for page in digital_result["pages"]:
        if not page["text_blocks"] and not page["tables"]:
            ocr_needed = True
            break
    
    # If no OCR needed, return digital result
    if not ocr_needed:
        logger.info("All pages successfully parsed digitally.")
        return digital_result
    
    # Otherwise, run OCR on the entire document
    logger.info("Some pages have no content from digital extraction. Running OCR...")
    ocr_result = parse_pdf_ocr(pdf_path, dpi=ocr_dpi)
    
    # Merge results - use OCR only for pages where digital extraction failed
    merged_result = {"pages": []}
    
    for digital_page in digital_result["pages"]:
        page_num = digital_page["page_num"]
        
        # If the page has content from digital extraction, use it
        if digital_page["text_blocks"] or digital_page["tables"]:
            merged_result["pages"].append(digital_page)
        else:
            # Find corresponding OCR page
            ocr_page = next((p for p in ocr_result["pages"] if p["page_num"] == page_num), None)
            if ocr_page:
                merged_result["pages"].append(ocr_page)
            else:
                # Fallback to empty page if OCR also failed
                merged_result["pages"].append(digital_page)
    
    return merged_result


def save_parsed_pdf_combined(pdf_path: str, output_path: str, ocr_dpi: int = 300) -> None:
    parsed_data = parse_pdf_combined(pdf_path, ocr_dpi)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(parsed_data, file, ensure_ascii=False, indent=2)
    
    logger.info(f"Combined parsed PDF saved to {output_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2:
        print("Usage: python combined.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_pdf_combined.json"
    ocr_dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    save_parsed_pdf_combined(pdf_path, output_path, ocr_dpi)