import os
import json
import logging
from typing import Dict, List, Any
import shutil
import tempfile
import shutil
import logging
from pypdf.errors import PdfReadWarning
import warnings
from camelot.io import read_pdf
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_camelot_temp():
    # Camelot writes out into your system temp dir (e.g. C:\Users\<you>\AppData\Local\Temp)
    tmp = tempfile.gettempdir()
    for name in os.listdir(tmp):
        path = os.path.join(tmp, name)
        # Camelot uses generic tmp names (tmpXXXXX), so this will catch most of them
        if name.startswith("tmp") and os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                # Windows often still has the file open; just skip those
                pass

# 1) Silence Camelot’s “no tables found” warnings
warnings.filterwarnings("ignore",
    category=UserWarning,
    message="No tables found in table area.*",
    module="camelot")

# 2) Silence the new deprecation warnings about rmtree’s parameters
warnings.filterwarnings("ignore",
    category=DeprecationWarning,
    module="shutil")

# 3) Monkey-patch shutil.rmtree to swallow any errors (using the new onexc API)
_orig_rmtree = shutil.rmtree
def safe_rmtree(path, *, ignore_errors=False, onexc=None):
    try:
        _orig_rmtree(path, ignore_errors=ignore_errors, onexc=onexc)
    except Exception:
        pass

shutil.rmtree = safe_rmtree

def extract_text_blocks(pdf_path: str, page_num: int) -> List[str]:
    """Extract text blocks from a specific page of a PDF file using PDFMiner."""
    text_blocks = []
    
    try:
        # Extract text using PDFMiner
        for page_layout in extract_pages(
            pdf_path, 
            page_numbers=[page_num-1],  # PDFMiner uses 0-based indexing
            laparams=LAParams(line_margin=0.5)
        ):
            # Extract text from each text element in the page
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    # Clean the text (remove excessive whitespace)
                    text = element.get_text().strip()
                    if text:  # Only add non-empty text blocks
                        text_blocks.append(text)
    except Exception as e:
        logger.error(f"Error extracting text from page {page_num}: {e}")
    
    return text_blocks


def extract_tables(pdf_path: str, page_num: int) -> List[List[List[str]]]:
    """Extract tables from a specific page of a PDF file using Camelot."""
    tables_data = []
    
    try:
        # Extract tables using Camelot
        # Try lattice mode first (for tables with borders)
        tables = read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
        
        # If no tables found with lattice, try stream mode (for tables without borders)
        if len(tables) == 0:
            tables = read_pdf(
                pdf_path, 
                pages=str(page_num), 
                flavor='stream',
                edge_tol=50,  # More tolerant of imperfections
                row_tol=10    # More tolerant of row variations
            )
        
        # Convert each table to a list of lists
        for table in tables:
            table_data = []
            df = table.df
            for i in range(len(df.index)):
                row = [str(cell).strip() for cell in df.iloc[i]]
                table_data.append(row)
            tables_data.append(table_data)
            
    except Exception as e:
        logger.error(f"Error extracting tables from page {page_num}: {e}")
    
    return tables_data


def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Parse a PDF file and extract text blocks and tables.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        A dictionary with the following structure:
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
    
    # Get the number of pages in the PDF
    with open(pdf_path, 'rb') as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)
        num_pages = sum(1 for _ in PDFPage.create_pages(document))
    
    # Process each page
    for page_num in range(1, num_pages + 1):
        page_data = {
            "page_num": page_num,
            "text_blocks": extract_text_blocks(pdf_path, page_num),
            "tables": extract_tables(pdf_path, page_num)
        }
        result["pages"].append(page_data)
    
    return result


def save_parsed_pdf(pdf_path: str, output_path: str) -> None:
    """Parse a PDF and save the result to a JSON file."""
    parsed_data = parse_pdf(pdf_path)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(parsed_data, file, ensure_ascii=False, indent=2)
    
    logger.info(f"Parsed PDF saved to {output_path}")

warnings.filterwarnings("ignore", category=PdfReadWarning)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_pdf.json"
    
    save_parsed_pdf(pdf_path, output_path)
    cleanup_camelot_temp()