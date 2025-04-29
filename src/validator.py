"""
Validator module for ensuring structured bank statement data
conforms to the expected schema.

This module validates transaction data and summary information,
converting types where possible and ensuring all required fields
are present with valid values.
"""

import re
import json
import logging
from datetime import datetime
from typing import Any, Dict

# Internal imports
from src.schema_loader import get_schema

# Configure logger
logger = logging.getLogger(__name__)

# Define allowed transaction categories
ALLOWED_CATEGORIES = {"deposit", "withdrawal", "withdrawal_regular_bill"}


class ValidationError(Exception):
    """Exception raised for validation errors that cannot be automatically fixed."""
    pass


def parse_date(date_str: str, statement_year=None) -> str:
    """
    Parse a date string in various formats and return standardized YYYY-MM-DD format.
    
    Args:
        date_str: Date string in various formats
        statement_year: Optional year to use for dates without years
        
    Returns:
        Date string in YYYY-MM-DD format
        
    Raises:
        ValidationError: If the date cannot be parsed
    """
    date_str = date_str.strip()
    
    # Common date formats to try
    formats = [
        "%Y-%m-%d",  # 2023-01-15
        "%m/%d/%Y",  # 01/15/2023
        "%d/%m/%Y",  # 15/01/2023
        "%m-%d-%Y",  # 01-15-2023
        "%d-%m-%Y",  # 15-01-2023
        "%b %d, %Y",  # Jan 15, 2023
        "%B %d, %Y",  # January 15, 2023
        "%d %b %Y",   # 15 Jan 2023
        "%d %B %Y",   # 15 January 2023
        "%m/%d/%y",   # 01/15/23
        "%d/%m/%y",   # 15/01/23
    ]
    
    # Try parsing with each format
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # Try formats without years if we have a statement year
    if statement_year:
        year_formats = [
            ("%d %b", "%d %b %Y"),      # 15 Jan → 15 Jan 2023
            ("%d %B", "%d %B %Y"),      # 15 January → 15 January 2023
            ("%b %d", "%b %d, %Y"),     # Jan 15 → Jan 15, 2023
            ("%B %d", "%B %d, %Y"),     # January 15 → January 15, 2023
            ("%d/%m", "%d/%m/%Y"),      # 15/01 → 15/01/2023
            ("%m/%d", "%m/%d/%Y"),      # 01/15 → 01/15/2023
            ("%d-%m", "%d-%m-%Y"),      # 15-01 → 15-01-2023
            ("%m-%d", "%m-%d-%Y"),      # 01-15 → 01-15-2023
        ]
        
        for short_fmt, full_fmt in year_formats:
            try:
                # Try parsing with short format to validate it's correct
                datetime.strptime(date_str, short_fmt)
                # If successful, add the year and parse with full format
                date_with_year = f"{date_str} {statement_year}"
                if "," in full_fmt:
                    date_with_year = f"{date_str}, {statement_year}"
                dt = datetime.strptime(date_with_year, full_fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    
    # If no format worked, try to extract date using regex
    # This is a fallback for unusual formats
    date_pattern = r"(\d{1,4})[-/\s](\d{1,2})[-/\s](\d{1,4})"
    match = re.search(date_pattern, date_str)
    
    if match:
        # Your existing regex handling code...
        parts = [match.group(1), match.group(2), match.group(3)]
        
        # Figure out which part is the year
        if int(parts[0]) > 1900 and int(parts[0]) <= datetime.now().year:
            # Format is likely YYYY-MM-DD
            try:
                dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        elif int(parts[2]) > 1900 and int(parts[2]) <= datetime.now().year:
            # Format is likely MM-DD-YYYY or DD-MM-YYYY
            try:
                # Try MM-DD-YYYY first
                dt = datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    # Try DD-MM-YYYY
                    dt = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
    
    # Check for just month and day without year (last resort)
    if statement_year and re.match(r"^\d{1,2}[/-]\d{1,2}$", date_str):
        parts = re.split(r"[/-]", date_str)
        # Try both MM/DD and DD/MM interpretations
        try:
            # Try as MM/DD
            dt = datetime(int(statement_year), int(parts[0]), int(parts[1]))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Try as DD/MM
                dt = datetime(int(statement_year), int(parts[1]), int(parts[0]))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    
    # If we still haven't parsed it, raise an error
    raise ValidationError(f"Could not parse date: '{date_str}'")


def parse_amount(amount_val: Any) -> float:
    """
    Parse an amount value into a standardized float.
    
    Args:
        amount_val: Amount value as string, float, or int
        
    Returns:
        Standardized float value
        
    Raises:
        ValidationError: If the amount cannot be parsed
    """
    if isinstance(amount_val, (int, float)):
        return float(amount_val)
    
    if not isinstance(amount_val, str):
        raise ValidationError(f"Amount must be a number or string, got {type(amount_val)}")
    
    # Remove currency symbols, commas, and other non-numeric characters
    # Keep negative signs, dots (decimal points)
    amount_str = amount_val.strip()
    
    # Special handling for parentheses indicating negative values
    if amount_str.startswith("(") and amount_str.endswith(")"):
        amount_str = amount_str.replace("(", "-").replace(")", "")
    
    # Remove currency symbols and separators
    amount_str = re.sub(r"[^\d.-]", "", amount_str)
    
    # Handle multiple decimal points (keep only the last one)
    parts = amount_str.split(".")
    if len(parts) > 2:
        amount_str = parts[0] + "." + "".join(parts[1:])
    
    try:
        return float(amount_str)
    except ValueError:
        raise ValidationError(f"Could not parse amount: '{amount_val}'")


def validate_transaction(txn: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean a single transaction against the schema.
    
    Args:
        txn: Transaction dictionary
        schema: Schema definition
        
    Returns:
        Cleaned transaction dictionary
        
    Raises:
        ValidationError: If the transaction cannot be validated
    """
    # Start with a new dict to ensure only valid fields are included
    validated_txn = {}
    
    # Use hardcoded categories instead of schema-based ones
    allowed_categories = ALLOWED_CATEGORIES
    default_category = "other"
    
    # Check required fields
    required_fields = ["date", "amount", "description"]
    for field in required_fields:
        if field not in txn:
            raise ValidationError(f"Transaction missing required field: {field}")
    
    # Validate and convert date (pass the statement_year)
    try:
        validated_txn["date"] = parse_date(txn["date"])
    except ValidationError as e:
        raise ValidationError(f"Invalid date format: {e}")
    
    # Rest of your validation code...
    # Validate and convert amount
    try:
        validated_txn["amount"] = parse_amount(txn["amount"])
    except ValidationError as e:
        raise ValidationError(f"Invalid amount format: {e}")
    
    # Validate description
    if not isinstance(txn["description"], str):
        validated_txn["description"] = str(txn["description"])
    else:
        validated_txn["description"] = txn["description"].strip()
    
    # Validate category
    if "category" in txn and txn["category"]:
        category = txn["category"].lower() if isinstance(txn["category"], str) else str(txn["category"]).lower()
        
        # Check if category is allowed
        if category in allowed_categories:
            validated_txn["category"] = category
        else:
            # Default to "other" if not allowed
            logger.warning(f"Invalid category '{category}', defaulting to '{default_category}'")
            validated_txn["category"] = default_category
    else:
        validated_txn["category"] = default_category
    
    # Include original row data if available
    if "original_row" in txn:
        validated_txn["original_row"] = txn["original_row"]
    
    return validated_txn


def validate_summary(summary: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the summary section against the schema.
    
    Args:
        summary: Summary dictionary
        schema: Schema definition
        
    Returns:
        Cleaned summary dictionary
    """
    # Start with a new dict to ensure only valid fields are included
    validated_summary = {}
    
    # Validate monthly summaries
    if "monthly_summaries" not in summary or not isinstance(summary["monthly_summaries"], list):
        logger.warning("Missing or invalid monthly_summaries, defaulting to empty list")
        validated_summary["monthly_summaries"] = []
    else:
        validated_summary["monthly_summaries"] = []
        
        for monthly in summary["monthly_summaries"]:
            if not isinstance(monthly, dict):
                logger.warning(f"Invalid monthly summary item: {monthly}")
                continue
                
            cleaned_monthly = {}
            
            # Validate month
            if "month" not in monthly or not monthly["month"]:
                logger.warning("Monthly summary missing month field, skipping")
                continue
                
            cleaned_monthly["month"] = monthly["month"]
            
            # Validate numeric fields - using correct keys
            numeric_fields = ["total_deposits", "total_withdrawals", "net_cash_flow"]
            for field in numeric_fields:
                if field in monthly:
                    try:
                        cleaned_monthly[field] = parse_amount(monthly[field])
                    except ValidationError:
                        cleaned_monthly[field] = 0.0
                        logger.warning(f"Invalid {field} in monthly summary, defaulting to 0")
                else:
                    cleaned_monthly[field] = 0.0
            
            # Add categories breakdown if available
            if "categories" in monthly and isinstance(monthly["categories"], dict):
                cleaned_monthly["categories"] = {}
                for cat, amount in monthly["categories"].items():
                    try:
                        cleaned_monthly["categories"][cat] = parse_amount(amount)
                    except ValidationError:
                        cleaned_monthly["categories"][cat] = 0.0
            else:
                cleaned_monthly["categories"] = {}
            
            validated_summary["monthly_summaries"].append(cleaned_monthly)
    
    # Validate overall summary
    if "overall_summary" not in summary or not isinstance(summary["overall_summary"], dict):
        logger.warning("Missing or invalid overall_summary, defaulting to empty dict")
        validated_summary["overall_summary"] = {
            "total_deposits": 0.0,
            "total_withdrawals": 0.0,
            "net_cash_flow": 0.0,
            "average_monthly_deposits": 0.0,
            "average_monthly_withdrawals": 0.0
        }
    else:
        cleaned_overall = {}
        
        # Validate numeric fields - using correct keys
        numeric_fields = [
            "total_deposits", "total_withdrawals", "net_cash_flow",
            "average_monthly_deposits", "average_monthly_withdrawals"
        ]
        
        for field in numeric_fields:
            if field in summary["overall_summary"]:
                try:
                    cleaned_overall[field] = parse_amount(summary["overall_summary"][field])
                except ValidationError:
                    cleaned_overall[field] = 0.0
                    logger.warning(f"Invalid {field} in overall summary, defaulting to 0")
            else:
                cleaned_overall[field] = 0.0
        
        # Add categories breakdown if available
        if "categories" in summary["overall_summary"] and isinstance(summary["overall_summary"]["categories"], dict):
            cleaned_overall["categories"] = {}
            for cat, amount in summary["overall_summary"]["categories"].items():
                try:
                    cleaned_overall["categories"][cat] = parse_amount(amount)
                except ValidationError:
                    cleaned_overall["categories"][cat] = 0.0
        else:
            cleaned_overall["categories"] = {}
        
        validated_summary["overall_summary"] = cleaned_overall
    
    # Validate recurring payments
    if "recurring_payments" not in summary or not isinstance(summary["recurring_payments"], list):
        logger.warning("Missing or invalid recurring_payments, defaulting to empty list")
        validated_summary["recurring_payments"] = []
    else:
        validated_summary["recurring_payments"] = []
        
        for payment in summary["recurring_payments"]:
            if not isinstance(payment, dict):
                continue
                
            cleaned_payment = {}
            
            # Required fields for recurring payments
            required = ["description", "frequency"]
            if not all(field in payment for field in required):
                logger.warning(f"Recurring payment missing required fields: {payment}")
                continue
            
            cleaned_payment["description"] = str(payment["description"])
            cleaned_payment["frequency"] = str(payment["frequency"])
            
            # Handle either amount or typical_amount
            amount_value = payment.get("amount", payment.get("typical_amount"))
            if amount_value:
                try:
                    cleaned_payment["amount"] = parse_amount(amount_value)
                except ValidationError:
                    logger.warning(f"Invalid amount in recurring payment: {payment}")
                    continue
            else:
                logger.warning(f"Missing amount in recurring payment: {payment}")
                continue
            
            # Optional fields
            if "category" in payment:
                cleaned_payment["category"] = str(payment["category"])
            
            if "dates" in payment and isinstance(payment["dates"], list):
                cleaned_payment["dates"] = payment["dates"]
            
            validated_summary["recurring_payments"].append(cleaned_payment)
    
    # Validate potential loans
    if "potential_loans" not in summary or not isinstance(summary["potential_loans"], list):
        logger.warning("Missing or invalid potential_loans, defaulting to empty list")
        validated_summary["potential_loans"] = []
    else:
        validated_summary["potential_loans"] = []
        
        for loan in summary["potential_loans"]:
            if not isinstance(loan, dict):
                continue
                
            cleaned_loan = {}
            
            # Required fields for potential loans - just description is required
            required = ["description"]
            if not all(field in loan for field in required):
                logger.warning(f"Potential loan missing required fields: {loan}")
                continue
            
            cleaned_loan["description"] = str(loan["description"])
            
            # Handle monthly_payment or payment_amount
            payment_value = loan.get("monthly_payment", loan.get("payment_amount"))
            if payment_value:
                try:
                    cleaned_loan["monthly_payment"] = parse_amount(payment_value)
                except ValidationError:
                    logger.warning(f"Invalid payment amount in potential loan: {loan}")
                    continue
            else:
                logger.warning(f"Missing payment amount in potential loan: {loan}")
                continue
            
            # Optional fields
            if "frequency" in loan:
                cleaned_loan["frequency"] = str(loan["frequency"])
            
            if "estimated_remaining_payments" in loan:
                try:
                    cleaned_loan["estimated_remaining_payments"] = int(loan["estimated_remaining_payments"])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid remaining payments in potential loan: {loan}")
            
            # Handle total_paid or estimated_total_amount
            total_value = loan.get("total_paid", loan.get("estimated_total_amount"))
            if total_value:
                try:
                    cleaned_loan["total_paid"] = parse_amount(total_value)
                except ValidationError:
                    logger.warning(f"Invalid total amount in potential loan: {loan}")
            
            validated_summary["potential_loans"].append(cleaned_loan)
    
    return validated_summary


def validate_structure(structured_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the complete structured data against the schema.
    
    Args:
        structured_data: Complete structured data (transactions and summary)
        schema: Schema definition
        
    Returns:
        Cleaned and validated structured data
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(structured_data, dict):
        raise ValidationError(f"Expected dict, got {type(structured_data)}")
    
    result = {}
    
    # Validate transactions
    if "transactions" not in structured_data:
        raise ValidationError("Missing required 'transactions' section")
    
    if not isinstance(structured_data["transactions"], list):
        raise ValidationError(f"Expected 'transactions' to be a list, got {type(structured_data['transactions'])}")
    
    result["transactions"] = []
    
    for i, txn in enumerate(structured_data["transactions"]):
        try:
            validated_txn = validate_transaction(txn, schema)
            result["transactions"].append(validated_txn)
        except ValidationError as e:
            logger.warning(f"Skipping invalid transaction at index {i}: {e}")
    
    if not result["transactions"]:
        raise ValidationError("No valid transactions found after validation")
    
    # Validate summary
    if "summary" not in structured_data:
        logger.warning("Missing 'summary' section, creating default")
        result["summary"] = {
            "monthly_summaries": [],
            "overall_summary": {
                "total_deposits": 0.0,
                "total_withdrawals": 0.0,
                "net_cash_flow": 0.0
            },
            "recurring_payments": [],
            "potential_loans": []
        }
    else:
        try:
            result["summary"] = validate_summary(structured_data["summary"], schema)
        except Exception as e:
            logger.warning(f"Error validating summary, creating default: {e}")
            result["summary"] = {
                "monthly_summaries": [],
                "overall_summary": {
                    "total_deposits": 0.0,
                    "total_withdrawals": 0.0,
                    "net_cash_flow": 0.0
                },
                "recurring_payments": [],
                "potential_loans": []
            }
    
    return result


def validate_output(structured_data: Dict[str, Any], bank_name: str) -> Dict[str, Any]:
    """
    Main validation function that loads schema and validates structured data.
    
    Args:
        structured_data: Complete structured data (transactions and summary)
        bank_name: Name of the bank to load appropriate schema
        
    Returns:
        Cleaned and validated structured data
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Load schema for the bank
        schema = get_schema(bank_name)
        
        # Validate structure against schema
        return validate_structure(structured_data, schema)
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise ValidationError(f"Failed to validate data: {str(e)}") from e


if __name__ == "__main__":
    """Test the validator with sample data"""
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 3:
        print("Usage: python validator.py <json_file> <bank_name>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    bank_name = sys.argv[2]
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        validated = validate_output(data, bank_name)
        print(json.dumps(validated, indent=2))
        print(f"Validation successful: {len(validated['transactions'])} transactions processed")
        
    except ValidationError as e:
        print(f"Validation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)