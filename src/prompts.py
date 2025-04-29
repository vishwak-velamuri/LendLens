"""
Prompt templates for LLM-based bank statement analysis.

This module contains template strings for three core tasks:
1. Finding transaction tables in raw document content
2. Mapping columns and categorizing transactions
3. Computing summary statistics from transaction data
"""

import json, sys
from typing import Dict, List, Any


def create_table_finder_prompt(
    raw_document: str,
    bank_name: str,
    table_markers: Dict[str, str] = None,
    max_content_length: int = sys.maxsize,
    column_definitions: List[Dict[str, Any]] = None
) -> str:
    # Safeguard against huge inputs
    if len(raw_document) > max_content_length:
        truncated_document = raw_document[:max_content_length] + "\n...(content truncated for length)"
    else:
        truncated_document = raw_document
    
    markers_str = ""
    if table_markers and table_markers.get("start") and table_markers.get("end"):
        markers_str = f"""
The transaction table typically starts with text matching "{table_markers['start']}" 
and ends with text matching "{table_markers['end']}".
"""
    
    # Add expected headers if provided
    expect_str = ""
    if column_definitions:
        header_names = [c.get('name', '') for c in column_definitions if c.get('name')]
        if header_names:
            expect_str = f"Expected headers (in any order): {header_names}"
    
    return f"""
# Bank Statement Transaction Table Extraction

## Task
Extract all transaction tables from this {bank_name} bank statement. Find the tables that contain 
financial transactions (deposits, withdrawals, payments, etc.).

{markers_str}

## Expected Columns
{expect_str}

## Instructions
1. Analyze the raw document content below.
2. Identify all tables that contain transaction data.
3. For each transaction table, extract its structure including headers and all rows.
4. Return the tables in JSON format.

## Output Format
Respond with a valid JSON object matching this structure:
```json
{{
  "tables": [
    {{
      "headers": [string, string, ...],    # list of column names
      "rows": [
        [string, string, ...],              # each transaction as a list of strings
        ...
      ]
    }}
    ...
  ]
}}
```
- Only include the tables actually extracted from the input.
- Do NOT invent, hallucinate, or copy any example data.
- Your output must strictly follow this JSON structure.
## Raw Document Content:
{truncated_document}

# Response (JSON only)
Do not include any explanation or preamble - respond with pure JSON only. Important: ignore any examples—extract tables only from the content below.
"""


def create_column_mapper_prompt(
    table: Dict[str, Any],
    bank_name: str,
    column_definitions: List[Dict[str, Any]],
    statement_period: str = None,
    max_transactions: int = sys.maxsize
) -> str:
    # Safeguard against huge inputs
    if len(table.get("rows", [])) > max_transactions:
        # Keep first 80% and last 20% of transactions to preserve most important data
        first_chunk = int(max_transactions * 0.8)
        last_chunk = max_transactions - first_chunk
        
        truncated_rows = table["rows"][:first_chunk] + \
                         table["rows"][-last_chunk:] if last_chunk > 0 else []
        
        truncated_table = {
            "headers": table["headers"],
            "rows": truncated_rows
        }
        
        transaction_note = f"Note: Transaction list truncated from {len(table['rows'])} to {len(truncated_rows)} for processing."
        table_json = json.dumps(truncated_table, indent=2)
    else:
        transaction_note = ""
        table_json = json.dumps(table, indent=2)
    
    # Format column definitions for the prompt
    column_defs_json = json.dumps(column_definitions, indent=2)
    
    # Create mappings directly from column names to canonical fields
    field_mappings = []
    for col in column_definitions:
        header = col.get("name", "")  # Use name instead of key (the visible header in PDF)
        col_name = header.lower()     # Use the header text for determining field type
        
        # Infer field mapping from name
        if "date" in col_name:
            field = "date"
        elif any(term in col_name for term in ["amount", "debit", "credit", "payment"]):
            field = "amount"
        elif any(term in col_name for term in ["description", "details", "transaction"]):
            field = "description"
        else:
            continue
            
        if header:
            field_mappings.append(f"{header} → {field}")
    
    mappings_str = "\n".join(field_mappings)
    period_context = ""
    if statement_period:
        period_context = f"""
## Statement Period Context
This statement covers the period: {statement_period}
When you see dates missing a year (like '04 Sep'), use the year from this statement period.
"""
    return f"""
# Transaction Column Mapping and Categorization

## Task
# UNIVERSAL RULES FOR TRANSACTION EXTRACTION (STRICT)
{period_context}
IMPORTANT:
- Use ONLY the input data provided (both tables and text blocks).
- NEVER hallucinate, invent, or add fake transactions.
- Process every meaningful transaction row from tables and text, if available.
- If a row is fully blank, you may skip it.

MULTI-SOURCE HANDLING:
- Some documents provide the same transaction information in both tables and text.
- You must merge information intelligently:
  - Prioritize structured tables over free text if available.
  - If information is missing in tables but available in nearby text, supplement it.
  - Avoid mapping the same transaction twice.

TABLE DETECTION:
- Process any block that appears structured into columns and rows.
- Ignore tables that are account details, regulatory footers, or ads.

COLUMN DETECTION:
- Column names will vary wildly. Match by purpose, not exact spelling.
- Detect:
  - Dates: "Date", "Transaction Date", "Posting Date", "Entry Date", etc.
  - Descriptions: "Description", "Details", "Transaction Info", "Narrative", etc.
  - Amounts:
    - Credits: "Credit", "In", "Deposit Amount", "Money In", "Received", etc.
    - Debits: "Debit", "Out", "Withdrawal", "Payment", "Money Out", "Spent", etc.

MERGING AMOUNTS:
- If two columns represent money coming in ("In") and money going out ("Out"), merge into one `amount` field.
- Credits must be POSITIVE amounts.
- Debits must be NEGATIVE amounts.
- If only one column exists ("Amount"), infer sign from context if possible (otherwise default to provided sign).

CURRENCY HANDLING:
- Strip all currency symbols (e.g., $, €, £, ₹, ¥).
- Preserve the numeric value without modification.
- No currency conversions.

DATE HANDLING:
- Normalize all dates to ISO 8601 format (YYYY-MM-DD).
- Supported input formats include:
  - "04 Sep19"
  - "09/04/2019"
  - "2019/09/04"
  - "September 4, 2019"
  - "4-9-19"
  - "4 Sept"
- If the date is missing the year:
  - Use the statement period shown elsewhere (e.g., "01 September 2019 to 30 November 2019") to infer the year.
- If day and month order is ambiguous:
  - Assume DD/MM/YYYY unless otherwise indicated by region (e.g., for US banks like Wells Fargo, assume MM/DD/YYYY).

DESCRIPTION HANDLING:
- Merge multi-line description fragments into a single description field.
- Remove excessive whitespace.
- Preserve important transaction metadata (invoice numbers, transaction IDs, etc.).

TRANSACTION TYPE MAPPING:
- Every transaction must be first categorized as either:
  - "deposit": Money coming into the account (positive amount)
  - "withdrawal": Money going out of the account (negative amount)

WITHDRAWAL SUBTAG:
- In addition to the basic deposit/withdrawal type, withdrawals can have this tag:
  - "regular_bill": For recurring payments like rent, utilities, subscriptions, loan payments, etc.
  - Only withdrawals (negative amounts) can have a regular_bill tag.
  - If a withdrawal is not a recurring bill (for example, a random shopping purchase), do not assign any subtag — just leave it as a plain withdrawal.
  - Deposits never get any subtags.

BASICALLY:
 - First categorize all transactions as either "deposit" or "withdrawal" based on whether money is coming in or going out.
 - Then, only for withdrawals, determine if they qualify as a "regular_bill" (for recurring payments like rent, utilities, etc.)
 - Deposits never get any subtags - they're just deposits.

GENERAL RULES:
- NEVER fabricate missing fields.
- If partial data is available (e.g., only amount and description, no date), still map it if meaningful.
- If a row contains only junk (e.g., advertisement text), skip it.
- Always normalize as best as possible based on available information.
- Preserve and pass through any special codes, references, or IDs included in descriptions.

OUTPUT RULES:
- Output must be strict, valid JSON.
- No markdown, no explanations, no formatting outside JSON.
- Exactly match the requested JSON structure.
- Respond only with the cleaned, mapped transaction list.

STRICTLY:
- No invented transactions.
- No assumed currencies or amounts.
- No hallucinated dates or descriptions.
- No duplicate mappings (one transaction only once except for withdrawals they can also be regular bills).


## Important: ignore any examples—map only the table I'm providing.
Map the columns in this {bank_name} bank statement table to standardized fields and categorize each transaction.
{transaction_note}

## Column Mapping Instructions
Use these column definitions from the {bank_name} schema to map table headers:
```json
{column_defs_json}
```

Key mappings:
{mappings_str}

## Data Normalization Requirements
1. Format dates as ISO format (YYYY-MM-DD)
2. Ensure amounts are numeric values (not strings)
3. For deposits (money coming in): use POSITIVE numbers
4. For withdrawals (money going out): use NEGATIVE numbers

## Transaction Categories
Categorize each transaction into one of these categories:
- deposit: Money coming into the account (positive amount)
- withdrawal: General money going out of the account (negative amount)
- withdrawal_regular_bill: Withdrawal that is a recurring payment (rent, utilities, subscriptions, loan payments, etc.)

## Input Table
```json
{table_json}
```

## Output Format
Respond with a valid JSON object matching this structure:
```json
{{
  "mapped_transactions": [
    {{
      "date": string,            # transaction date in "YYYY-MM-DD" format
      "amount": number,          # positive for deposits, negative for withdrawals
      "description": string,     # transaction description
      "category": string,        # one of: deposit, withdrawal, withdrawal_regular_bill
      "original_row": [string, ...]  # the original raw row data as a list of strings
    }},
    ...
  ]
}}
```
- Only map transactions based on the input table provided.
- Do NOT invent, hallucinate, or copy any example data.
- Your output must strictly follow this JSON structure.
# Response (JSON only)
Do not include any explanation or preamble - respond with pure JSON only. Important: ignore any examples—map only the table I'm providing.
"""


def create_summary_prompt(
    mapped_transactions: List[Dict[str, Any]], 
    max_transactions: int = 300
) -> str:
    # Safeguard against huge inputs
    if len(mapped_transactions) > max_transactions:
        # Keep first 70% and last 30% of transactions to preserve most important data
        first_chunk = int(max_transactions * 0.7)
        last_chunk = max_transactions - first_chunk
        
        truncated_transactions = mapped_transactions[:first_chunk] + \
                                mapped_transactions[-last_chunk:] if last_chunk > 0 else []
        
        transaction_note = f"Note: Transaction list truncated from {len(mapped_transactions)} to {len(truncated_transactions)} for processing."
    else:
        truncated_transactions = mapped_transactions
        transaction_note = ""
    
    transactions_json = json.dumps({"mapped_transactions": truncated_transactions}, indent=2)
    
    return f"""
# Transaction Summary Computation
## TASK: Respond with pure JSON only.  Do NOT include any explanation, markdown, or examples.
## Task
## Important: ignore any examples—summarize only the transactions I'm providing.
Analyze the provided transactions and compute monthly summary statistics.
{transaction_note}

## Instructions
1. Group transactions by month.
2. For each month, calculate:
   - Total deposits (sum of amounts where amount > 0)
   - Total withdrawals (sum of amounts where amount < 0)
   - Total regular bills (sum of amounts for withdrawal_regular_bill category)
3. Calculate overall totals across all months.
4. Identify recurring payments (these are the same as regular bills - transactions with category 'withdrawal_regular_bill').
5. Look for potential loan payments (consistent monthly outgoing payments).

## Important Note on Field Names
- Use "total_regular_bills" for regular bill totals (NOT "regular_bill_total")
- Ensure amounts for withdrawals and regular bills are negative numbers

## Input Transactions
```json
{transactions_json}
```

## Output Format
Return a JSON object with this structure:
```json
{{
  "monthly_summaries": [
    {{
      "month": string,             # month in "YYYY-MM" format
      "total_deposits": number,     # sum of all deposits in that month
      "total_withdrawals": number,  # sum of all withdrawals in that month
      "total_regular_bills": number, # sum of all regular bills in that month (subset of withdrawals)
      "net_cash_flow": number,      # total_deposits + total_withdrawals
      "categories": {{
        "deposit": number,
        "withdrawal": number,
        "withdrawal_regular_bill": number
      }}
    }}
  ],
  "overall_summary": {{
    "total_deposits": number,
    "total_withdrawals": number,
    "total_regular_bills": number,  # This is a subset of total_withdrawals
    "net_cash_flow": number,
    "average_monthly_deposits": number,
    "average_monthly_withdrawals": number,
    "categories": {{
      "deposit": number,
      "withdrawal": number,
      "withdrawal_regular_bill": number
    }}
  }},
  "recurring_payments": [
    {{
      "description": string,      # description of recurring payment (same as regular bill)
      "amount": number,            # recurring payment amount (negative)
      "frequency": string          # e.g., "monthly"
    }},
    ...
  ],
  "potential_loans": [
    {{
      "description": string,      # description of the loan
      "monthly_payment": number,  # monthly payment amount (negative)
      "total_paid": number         # total amount paid so far
    }},
    ...
  ]
}}
```
- Only compute summaries based on the provided mapped transactions.
- Do NOT invent, hallucinate, or copy any example data.
- Your output must strictly match this structure and type definitions.
- Note that recurring payments should directly correspond to transactions categorized as withdrawal_regular_bill.
# Response (JSON only)
Do not include any explanation or preamble - respond with pure JSON only. Important: ignore any examples—summarize only the transactions I'm providing.
"""


def create_validation_prompt(output_json: str, schema: Dict[str, Any]) -> str:
    return f"""
# Validate JSON Output

## Task
Given this JSON:
```json
{output_json}
```

and this bank-specific schema definition (not a standard JSON Schema):
```json
{json.dumps(schema, indent=2)}
```

Validate that the output conforms to the expected structure. If any fields are missing, mis-typed, or extra, correct them.

## Instructions
1. Check that all required fields exist
2. Verify data types match the expected types
3. Remove any fields not in the expected structure
4. Add missing fields with default values if needed
5. Format numbers correctly (e.g., ensure amounts are numbers, not strings)
6. Ensure all category values match exactly the expected values (deposit, withdrawal, withdrawal_regular_bill)

# Response (JSON only)
Return only the corrected JSON without any explanation. Important: ignore any examples.
"""