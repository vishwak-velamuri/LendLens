import re
import json
import time
from typing import Any, Dict, List, TypedDict

import logging
logger = logging.getLogger(__name__)

from src.combined import parse_pdf_combined
from src.schema_loader import get_columns, get_table_markers, get_schema
from src.prompts import (
    create_table_finder_prompt,
    create_column_mapper_prompt,
    create_summary_prompt
)
from src.llm_client import call_llm

class TableDict(TypedDict):
    headers: List[str]
    rows: List[List[str]]

class MappedTxn(TypedDict):
    date: str
    amount: float
    description: str
    category: str
    original_row: List[str]

class SummaryDict(TypedDict):
    monthly_summaries: List[Dict]
    overall_summary: Dict
    recurring_payments: List[Dict]
    potential_loans: List[Dict]

class StructuringError(Exception):
    pass


def extract_tables(parsed_doc: dict, bank_name: str) -> List[TableDict]:
    start_time = time.time()
    logger.info(f"Extracting tables for {bank_name} statement...")
    
    try:
        # Convert parsed document to a string representation
        raw_str = json.dumps(parsed_doc, indent=None)
        with open("raw_input_to_llm.json", "w", encoding="utf-8") as f:
            f.write(raw_str)
        # Load schema information
        markers = get_table_markers(bank_name)
        columns = get_columns(bank_name)
        
        # Create prompt and call LLM
        prompt = create_table_finder_prompt(
            raw_str, 
            bank_name, 
            markers, 
            max_content_length=8000,
            column_definitions=columns
        )
        logger.debug("=== LLM Prompt ===\n" + prompt + "\n=== End Prompt ===")
        llm_response = call_llm(prompt)
        logger.debug("=== LLM Response ===\n" + llm_response + "\n=== End Response ===")
        
        # Parse LLM response to extract tables
        try:
            result = json.loads(llm_response)
            tables = result.get("tables", [])
            
            if not tables:
                logger.warning("No tables found in the document")
            else:
                logger.info(f"Found {len(tables)} potential transaction tables")
                
            return tables
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"LLM response was: {llm_response}")
            raise StructuringError(f"Failed to extract tables: Invalid JSON response from LLM") from e
            
    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        raise StructuringError(f"Failed to extract tables: {str(e)}") from e
    finally:
        logger.info(f"Table extraction took {time.time() - start_time:.2f} seconds")


def map_transactions(table: TableDict, bank_name: str, parsed_doc: dict = None) -> List[MappedTxn]:
    start_time = time.time()
    logger.info(f"Mapping transactions for {bank_name}...")
    
    try:
        # Load column definitions
        columns = get_columns(bank_name)
        statement_period = None
        if parsed_doc:
            for page in parsed_doc.get("pages", []):
                for block in page.get("text_blocks", []):
                    # block is a string, not a dict
                    if re.search(r"(statement|period).*?\d+.*?to.*?\d+", block, re.IGNORECASE):
                        statement_period = block
                        break
        # Create prompt and call LLM
        prompt = create_column_mapper_prompt(table, bank_name, columns, statement_period)
        logger.debug("=== LLM Prompt ===\n" + prompt + "\n=== End Prompt ===")
        llm_response = call_llm(prompt)
        logger.debug("=== LLM Response ===\n" + llm_response + "\n=== End Response ===")
        
        # Parse LLM response
        try:
            result = json.loads(llm_response)
            mapped_transactions = result.get("mapped_transactions", [])
            
            if not mapped_transactions:
                logger.warning("No transactions mapped")
            else:
                logger.info(f"Mapped {len(mapped_transactions)} transactions")
                
            return mapped_transactions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"LLM response was: {llm_response}")
            raise StructuringError(f"Failed to map transactions: Invalid JSON response from LLM") from e
            
    except Exception as e:
        logger.error(f"Error mapping transactions: {str(e)}")
        raise StructuringError(f"Failed to map transactions: {str(e)}") from e
    finally:
        logger.info(f"Transaction mapping took {time.time() - start_time:.2f} seconds")


def summarize_transactions(mapped: List[MappedTxn]) -> SummaryDict:
    start_time = time.time()
    logger.info("Generating transaction summary...")
    
    try:
        # Create prompt and call LLM
        prompt = create_summary_prompt(mapped)
        llm_response = call_llm(prompt)
        logger.debug("=== Summary LLM Response START ===\n" + llm_response + "\n=== Summary LLM Response END ===")
        
        # Parse LLM response
        try:
            summary = json.loads(llm_response)
            
            # Basic validation and field normalization
            required_keys = ["monthly_summaries", "overall_summary", "recurring_payments", "potential_loans"]
            for key in required_keys:
                if key not in summary:
                    logger.warning(f"Summary missing required key: {key}")
                    summary[key] = [] if key != "overall_summary" else {}
            
            # Ensure overall_summary has all required fields
            overall = summary.get("overall_summary", {})
            required_overall_fields = [
                "total_deposits", "total_withdrawals", "total_regular_bills", 
                "net_cash_flow"
            ]
            
            for field in required_overall_fields:
                if field not in overall:
                    logger.warning(f"Overall summary missing required field: {field}")
                    overall[field] = 0.0
            
            # Ensure we're using the correct field name for regular bills
            if "regular_bill_total" in overall and "total_regular_bills" not in overall:
                overall["total_regular_bills"] = overall["regular_bill_total"]
                
            # Ensure recurring payments are treated as regular bills
            recurring_payments = summary.get("recurring_payments", [])
            if recurring_payments:
                # Add up all recurring payment amounts
                recurring_total = sum(abs(payment.get("amount", 0)) for payment in recurring_payments)
                
                # If regular bills total is missing or zero but we have recurring payments,
                # use the recurring payment total
                if overall.get("total_regular_bills", 0) == 0 and recurring_total > 0:
                    overall["total_regular_bills"] = -recurring_total
            
            # Ensure categories dict exists with our three categories
            if "categories" not in overall:
                overall["categories"] = {
                    "deposit": 0.0,
                    "withdrawal": 0.0,
                    "withdrawal_regular_bill": 0.0
                }
            
            # Ensure only our three allowed categories exist
            categories = overall.get("categories", {})
            allowed_categories = ["deposit", "withdrawal", "withdrawal_regular_bill"]
            for category in list(categories.keys()):
                if category not in allowed_categories:
                    del categories[category]
            
            for category in allowed_categories:
                if category not in categories:
                    categories[category] = 0.0
                    
            # Ensure negative values for withdrawals and regular bills
            if overall.get("total_withdrawals", 0) > 0:
                overall["total_withdrawals"] = -abs(overall["total_withdrawals"])
                
            if overall.get("total_regular_bills", 0) > 0:
                overall["total_regular_bills"] = -abs(overall["total_regular_bills"])
            
            return summary
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"LLM response was: {llm_response}")
            raise StructuringError("Failed to summarize transactions: Invalid JSON response from LLM") from e
            
    except Exception as e:
        logger.error(f"Error summarizing transactions: {str(e)}")
        raise StructuringError(f"Failed to summarize transactions: {str(e)}") from e
    finally:
        logger.info(f"Transaction summarization took {time.time() - start_time:.2f} seconds")


def validate_output(structured: Any, bank_name: str) -> Dict[str, Any]:
    start_time = time.time()
    logger.info("Validating structured output (python)…")
    try:
        # Load the bank-specific schema
        schema = get_schema(bank_name)
        # Import and run the Python-based validator
        from src.validator import validate_structure

        cleaned = validate_structure(structured, schema)
        logger.info("Python validation succeeded")
        return cleaned

    except Exception as e:
        logger.error(f"Python validation failed, returning original data: {e}")
        return structured

    finally:
        logger.info(f"Output validation took {time.time() - start_time:.2f} seconds")


def analyze_statement(pdf_path: str, bank_name: str) -> Dict:
    start_time = time.time()
    logger.info(f"Starting analysis of {pdf_path} for bank: {bank_name}")
    
    try:
        # Step 1: Parse PDF
        parsed = parse_pdf_combined(pdf_path)
        logger.info("PDF parsed successfully")
        
        # Step 2: Extract tables
        tables = extract_tables(parsed, bank_name)
        if not tables:
            raise StructuringError("No tables found in the document")
        
        # Step 3: Select transaction table (assuming the one with the most rows is the main table)
        tables.sort(key=lambda t: len(t["rows"]), reverse=True)
        chosen_table = tables[0]
        logger.info(f"Selected table with {len(chosen_table['rows'])} rows")
        
        # Step 4: Map transactions
        mapped = map_transactions(chosen_table, bank_name, parsed)
        if not mapped:
            raise StructuringError("No transactions mapped from the table")
        
        # Step 5: Summarize transactions
        summary = summarize_transactions(mapped)
        
        # Step 6: Validate output
        result = {
            "transactions": mapped,
            "summary": summary
        }
        
        validated = validate_output(result, bank_name)
        
        logger.info(f"Analysis completed successfully in {time.time() - start_time:.2f} seconds")
        return validated
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Analysis failed after {elapsed:.2f} seconds: {str(e)}")
        
        if isinstance(e, StructuringError):
            raise
        else:
            raise StructuringError(f"Statement analysis failed: {str(e)}") from e


if __name__ == "__main__":
    import sys
    import os
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 3:
        print("Usage: python structurer.py <pdf_path> <bank_name>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    bank_name = sys.argv[2]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    try:
        result = analyze_statement(pdf_path, bank_name)
        print(json.dumps(result, indent=2))
    except StructuringError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)