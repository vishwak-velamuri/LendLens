import numpy as np
import pandas as pd
from typing import Dict, Any, List
from functools import lru_cache


# Cache the feature names to avoid re-reading the CSV on every call
@lru_cache(maxsize=1)
def get_feature_names():
    try:
        return pd.read_csv("data/processed/feature_names.csv", header=None).iloc[:, 0].tolist()
    except FileNotFoundError:
        # Fallback if file isn't available
        return [
            "num_months", "total_deposits", "total_withdrawals", "total_withdrawal_regular_bill",
            "net_cash_flow", "deposit", "withdrawal", "withdrawal_regular_bill",
            "potential_loan_count", "potential_loan_total_paid"
        ]


def compute_features(structured: Dict[str, Any], form: Dict[str, float]) -> np.ndarray:
    # Get cached feature names to ensure correct ordering
    feature_names = get_feature_names()
    
    # Extract data from structured JSON
    summary = structured.get("summary", {})
    monthly_summaries = summary.get("monthly_summaries", [])
    overall_summary = summary.get("overall_summary", {})
    potential_loans = summary.get("potential_loans", [])
    
    # Count of months with data
    num_months = len(monthly_summaries)
    
    # Initialize variables for aggregation
    monthly_deposits = []
    monthly_withdrawals = []
    monthly_regular_bill = []
    
    # Process monthly data
    for month in monthly_summaries:
        monthly_deposits.append(month.get("total_deposits", 0.0))
        monthly_withdrawals.append(month.get("total_withdrawals", 0.0))
        
        # Extract regular bills from the categories map
        categories = month.get("categories", {})
        regular_bill = categories.get("withdrawal_regular_bill", 0.0)
        monthly_regular_bill.append(regular_bill)
    
    # Calculate total values
    total_deposits = sum(monthly_deposits)
    total_withdrawals = sum(monthly_withdrawals)
    total_withdrawal_regular_bill = sum(monthly_regular_bill)
    
    # Get net cash flow directly from overall summary or calculate it
    net_cash_flow = overall_summary.get("net_cash_flow", total_deposits + total_withdrawals)
    
    # Calculate per-month averages
    deposit = total_deposits / num_months if num_months > 0 else 0.0
    withdrawal = abs(total_withdrawals) / num_months if num_months > 0 else 0.0
    withdrawal_regular_bill = total_withdrawal_regular_bill / num_months if num_months > 0 else 0.0
    
    # Potential loan metrics
    potential_loan_count = len(potential_loans)
    
    # Use total_paid instead of monthly_payment to match training data
    potential_loan_total_paid = sum(abs(loan.get("total_paid", 0.0)) for loan in potential_loans)
    
    # Create the feature map for programmatic ordering
    feature_map = {
        "num_months": num_months,
        "total_deposits": total_deposits,
        "total_withdrawals": total_withdrawals,
        "total_withdrawal_regular_bill": total_withdrawal_regular_bill,
        "net_cash_flow": net_cash_flow,
        "deposit": deposit,
        "withdrawal": withdrawal,
        "withdrawal_regular_bill": withdrawal_regular_bill,
        "potential_loan_count": potential_loan_count,
        "potential_loan_total_paid": potential_loan_total_paid
    }
    
    # Create feature vector in correct order
    feature_vector = np.array([feature_map[name] for name in feature_names], dtype=float)
    
    return feature_vector

if __name__ == "__main__":
    import json
    from src.feature_computation import compute_features, get_feature_names

    # 1) Load a sample structured JSON file
    sample_path = "data/structured/Commonwealth.json"
    structured = json.load(open(sample_path, "r"))

    # 2) Define a dummy form
    form = {
        "loan_amount": 25000.0,
        "down_payment": 5000.0,
        "interest_rate": 4.5,
        "term_months": 48
    }

    # 3) Compute
    feature_names = get_feature_names()
    vector = compute_features(structured, form)

    # 4) Print results
    print("Feature names (in order):")
    for name in feature_names:
        print(f"  - {name}")
    print("\nComputed feature vector:")
    print(vector)
    print("\nVector length:", len(vector))
