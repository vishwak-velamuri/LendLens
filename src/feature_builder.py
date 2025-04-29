import os
import json
import glob
import pandas as pd
import numpy as np
import logging
import argparse
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build features from financial statements")
    parser.add_argument("--data-dir", default="data/structured", 
                        help="Directory containing structured JSON statements")
    parser.add_argument("--labels-file", default="data/labels.csv",
                        help="CSV file containing approval labels")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directory to save processed features")
    parser.add_argument("--output-format", choices=["csv", "numpy", "both"], default="both",
                        help="Format to save features (csv, numpy, or both)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def normalize_filename(filename: str) -> str:
    # Extract basename, remove extension, lowercase
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0].lower()
    return base


def discover_files(data_dir: str, labels_file: str) -> Tuple[List[str], Dict[str, int]]:
    # Find all statement files
    json_pattern = os.path.join(data_dir, "*.json")
    json_files = glob.glob(json_pattern)
    logger.info(f"Discovered {len(json_files)} statement files from {json_pattern}")
    
    # Load labels from CSV
    try:
        labels_df = pd.read_csv(labels_file)
        logger.info(f"Loaded {len(labels_df)} labels from {labels_file}")
    except Exception as e:
        logger.error(f"Failed to load labels from {labels_file}: {e}")
        raise
    
    # Create normalized filename to label mapping
    filename_to_label = {}
    for _, row in labels_df.iterrows():
        # Normalize the filename from labels.csv
        norm_filename = normalize_filename(row['filename'])
        filename_to_label[norm_filename] = int(row['approved'])
    
    return json_files, filename_to_label


def load_statement(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def calculate_category_averages(transactions_df: pd.DataFrame, num_months: int) -> Dict[str, float]:
    categories = {
        'deposit': 0.0,
        'withdrawal': 0.0,
        'withdrawal_regular_bill': 0.0
    }
    
    # Default to 1 month if num_months is 0 or not provided
    safe_num_months = max(1, num_months) if num_months else 1
    
    # If there's no category column, try to infer from amount
    if 'category' not in transactions_df.columns or transactions_df.empty:
        # If there's an amount column, use sign to determine deposit vs withdrawal
        if 'amount' in transactions_df.columns:
            deposits = transactions_df[transactions_df['amount'] > 0]['amount'].sum()
            withdrawals = transactions_df[transactions_df['amount'] < 0]['amount'].sum()
            
            categories['deposit'] = float(deposits) / safe_num_months if not pd.isna(deposits) else 0.0
            categories['withdrawal'] = float(abs(withdrawals)) / safe_num_months if not pd.isna(withdrawals) else 0.0
        return categories
    
    # Group by category and sum amounts
    category_totals = transactions_df.groupby('category')['amount'].sum()
    
    # Fill in our predefined categories
    for category in categories:
        if category in category_totals:
            categories[category] = float(abs(category_totals[category])) / safe_num_months
    
    return categories


def calculate_monthly_flow_metrics(transactions_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        'num_months': 1,  # Default to 1 to avoid division by zero
    }
    
    # Check if date column exists
    if 'date' not in transactions_df.columns or transactions_df.empty:
        return metrics
    
    try:
        # Convert date to datetime and extract month
        transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
        
        # Drop rows where date conversion failed
        valid_dates_df = transactions_df.dropna(subset=['date'])
        
        if not valid_dates_df.empty:
            transactions_df['month'] = valid_dates_df['date'].dt.to_period('M')
            
            # Count number of unique months
            unique_months = len(transactions_df['month'].unique())
            metrics['num_months'] = max(1, unique_months)  # Ensure at least 1 month
    except Exception as e:
        logger.warning(f"Error calculating monthly flows: {e}")
        
    return metrics


def calculate_potential_loans(statement: Dict[str, Any]) -> Dict[str, float]:
    potential_loans = statement.get('summary', {}).get('potential_loans', [])
    
    return {
        'potential_loan_count': len(potential_loans),
        'potential_loan_total_paid': sum(abs(float(loan.get('total_paid', 0))) for loan in potential_loans)
    }


def extract_features(statement: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    
    # Get transactions and summary
    transactions = (statement.get('mapped_transactions') or statement.get('transactions', []))
    summary = statement.get('summary', {})
    
    # Convert transactions to DataFrame for easier processing
    try:
        transactions_df = pd.DataFrame(transactions)
        # Ensure amount is numeric
        if 'amount' in transactions_df.columns:
            transactions_df['amount'] = transactions_df['amount'].astype(float)
        else:
            transactions_df['amount'] = 0.0
            logger.warning("Transactions missing 'amount' field")
    except Exception as e:
        logger.warning(f"Error converting transactions to DataFrame: {e}")
        transactions_df = pd.DataFrame(columns=['amount', 'category', 'date', 'description'])
    
    # Get monthly statistics
    monthly_metrics = calculate_monthly_flow_metrics(transactions_df)
    features.update(monthly_metrics)
    
    if 'category' in transactions_df.columns and 'withdrawal_regular_bill' in transactions_df['category'].unique():
        bill_mask = transactions_df['category'] == 'withdrawal_regular_bill'
        features['total_withdrawal_regular_bill'] = float(transactions_df.loc[bill_mask, 'amount'].abs().sum())
    else:
        # …otherwise fall back to the summary’s category total
        cats = summary.get('overall_summary', {}).get('categories', {})
        features['total_withdrawal_regular_bill'] = abs(cats.get('withdrawal_regular_bill', 0))

    # If num_months is 0, set it to 1 to avoid division by zero
    if features['num_months'] == 0:
        features['num_months'] = 1
    
    # Overall totals from summary - correctly navigate the nested structure
    overall = summary.get('overall_summary', {})
    features['total_deposits'] = float(overall.get('total_deposits', 0))
    features['total_withdrawals'] = float(overall.get('total_withdrawals', 0))
    features['net_cash_flow'] = float(overall.get('net_cash_flow', 0))
    
    # Calculate monthly averages using the absolute amounts from the totals 
    # (we don't need the category averages function here, we can calculate directly)
    features['deposit'] = features['total_deposits'] / features['num_months']
    features['withdrawal'] = abs(features['total_withdrawals']) / features['num_months']
    features['withdrawal_regular_bill'] = (features['total_withdrawal_regular_bill'] / features['num_months'])
    
    # Potential loan metrics
    loan_metrics = calculate_potential_loans(statement)
    features.update(loan_metrics)
        
    return features


def build_features(data_dir: str, labels_file: str) -> pd.DataFrame:
    # Discover files and load labels
    json_files, filename_to_label = discover_files(data_dir, labels_file)
    
    # Process each statement
    all_features = []
    unmatched_files = []
    
    for file_path in json_files:
        try:
            # Load statement
            statement = load_statement(file_path)
            
            # Extract features
            features = extract_features(statement)
            
            # Add filename
            features['filename'] = os.path.basename(file_path)
            
            # Add label using normalized filename
            norm_filename = normalize_filename(file_path)
            if norm_filename in filename_to_label:
                features['approved'] = filename_to_label[norm_filename]
            else:
                features['approved'] = None
                unmatched_files.append(file_path)
            
            all_features.append(features)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Log unmatched files
    if unmatched_files:
        logger.warning(f"{len(unmatched_files)} files had no matching label in {labels_file}:")
        for file in unmatched_files[:5]:  # Log first 5 unmatched files
            logger.warning(f"  - {file}")
        if len(unmatched_files) > 5:
            logger.warning(f"  - ... and {len(unmatched_files) - 5} more")
        
        # Filter out rows with missing labels
        df = df.dropna(subset=['approved'])
        logger.info(f"Dropped {len(unmatched_files)} rows with missing labels")
    
    return df


def save_features(df: pd.DataFrame, output_dir: str, output_format: str = 'both') -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate features from labels
    X = df.drop(['filename', 'approved'], axis=1)
    y = df['approved']
    
    if output_format in ['csv', 'both']:
        csv_path = os.path.join(output_dir, 'features.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved features to {csv_path}")
    
    if output_format in ['numpy', 'both']:
        x_path = os.path.join(output_dir, 'X.npy')
        y_path = os.path.join(output_dir, 'y.npy')
        features_path = os.path.join(output_dir, 'feature_names.csv')
        
        np.save(x_path, X.values)
        np.save(y_path, y.values)
        
        # Save column names for reference
        pd.Series(X.columns).to_csv(features_path, index=False, header=False)
        
        logger.info(f"Saved features to {x_path} and {y_path}")
        logger.info(f"Saved feature names to {features_path}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting feature extraction process...")
    
    # Build features
    df = build_features(args.data_dir, args.labels_file)
    
    # Print statistics
    logger.info(f"Extracted features from {len(df)} statements")
    logger.info(f"Approval rate: {df['approved'].mean():.2f}")
    logger.info(f"Number of features: {df.shape[1] - 2}")  # Subtract filename and approved columns
    
    # Display feature names and their mean values
    feature_means = df.drop(['filename', 'approved'], axis=1).mean().sort_values(ascending=False)
    logger.info("Top 5 features by mean value:")
    for feature, value in feature_means.head().items():
        logger.info(f"  - {feature}: {value:.2f}")
    
    # Save features
    save_features(df, args.output_dir, args.output_format)
    
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()