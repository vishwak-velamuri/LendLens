import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import argparse
import joblib
from typing import Dict, Tuple, Any

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost model")
    parser.add_argument("--input-dir", default="data/processed",
                        help="Directory containing processed features and model")
    parser.add_argument("--model-path", default=None,
                        help="Path to trained model (default: input-dir/model.joblib)")
    parser.add_argument("--output-path", default=None,
                        help="Path to save evaluation results (default: input-dir/evaluation_results.json)")
    parser.add_argument("--input-format", default="auto", choices=["auto", "numpy", "csv"],
                        help="Format of input data (auto will try numpy first, then csv)")
    return parser.parse_args()


def load_data(input_dir: str, input_format: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    # Try numpy format first if auto or numpy is specified
    if input_format in ["auto", "numpy"]:
        try:
            X_path = os.path.join(input_dir, "X.npy")
            y_path = os.path.join(input_dir, "y.npy")
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                logger.info(f"Loading data from numpy files: {X_path}, {y_path}")
                X = np.load(X_path)
                y = np.load(y_path)
                
                # Load feature names if available
                feature_names_path = os.path.join(input_dir, "feature_names.csv")
                if os.path.exists(feature_names_path):
                    feature_names = pd.read_csv(feature_names_path, header=None).iloc[:, 0].tolist()
                    logger.info(f"Loaded {len(feature_names)} feature names")
                
                return X, y
        except Exception as e:
            if input_format == "numpy":
                raise
            logger.warning(f"Failed to load numpy data: {e}")
    
    # Fall back to CSV if auto or explicitly specified
    if input_format in ["auto", "csv"]:
        try:
            csv_path = os.path.join(input_dir, "features.csv")
            logger.info(f"Loading data from CSV: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # Assume last column is the target if not named 'approved'
            target_col = 'approved' if 'approved' in df.columns else df.columns[-1]
            
            # Skip non-feature columns
            non_feature_cols = ['filename', target_col]
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            logger.info(f"Loaded {X.shape[1]} features and {len(y)} samples")
            return X, y
            
        except Exception as e:
            if input_format == "csv":
                raise
            logger.error(f"Failed to load CSV data: {e}")
    
    raise ValueError(f"Could not load data using format: {input_format}")


def load_model(model_path: str) -> Any:
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    logger.info("Evaluating model performance")
    
    # Get predictions
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1]
    
    # Initialize results dictionary
    results = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "class_distribution": {
            "actual": np.bincount(y.astype(int)).tolist(),
            "predicted": np.bincount(y_pred.astype(int)).tolist()
        }
    }
    
    # Calculate ROC AUC if we have both classes
    unique_classes = np.unique(y)
    if len(unique_classes) > 1:
        results["roc_auc"] = float(roc_auc_score(y, y_probs))
    else:
        logger.warning(f"Cannot calculate ROC AUC: only found classes {unique_classes}")
        results["roc_auc"] = None
    
    # Get classification report as dictionary
    report = classification_report(y, y_pred, output_dict=True)
    processed_report: Dict[str, Any] = {}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            # per‐class or average metrics → map each to float
            processed_report[label] = {
                mname: float(mval) for mname, mval in metrics.items()
            }
        else:
            # the "accuracy" entry is a scalar
            processed_report[label] = float(metrics)
    results["classification_report"] = processed_report
    
    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    logger.info("========== MODEL EVALUATION RESULTS ==========")
    
    # Print accuracy and ROC AUC
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    if results["roc_auc"] is not None:
        logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
    else:
        logger.info("ROC AUC: Not available (insufficient class variety)")
    
    # Print confusion matrix
    cm = np.array(results["confusion_matrix"])
    logger.info("Confusion Matrix:")
    logger.info(f"               Predicted Negative    Predicted Positive")
    logger.info(f"Actual Negative    {cm[0][0]:<20d}{cm[0][1]}")
    logger.info(f"Actual Positive    {cm[1][0]:<20d}{cm[1][1]}")
    
    # Print class distribution
    logger.info("Class Distribution:")
    logger.info(f"  Actual:    {results['class_distribution']['actual']}")
    logger.info(f"  Predicted: {results['class_distribution']['predicted']}")
    
    # Print classification report metrics for each class
    logger.info("Classification Report:")
    for class_name in ['0', '1']:
        if class_name in results["classification_report"]:
            metrics = results["classification_report"][class_name]
            logger.info(f"  Class {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")
            logger.info(f"    Support:   {metrics['support']}")
    
    # Print overall metrics
    if 'macro avg' in results["classification_report"]:
        logger.info("  Macro Average:")
        for k, v in results["classification_report"]["macro avg"].items():
            if isinstance(v, (int, float)):
                logger.info(f"    {k}: {v:.4f}")
    
    if 'weighted avg' in results["classification_report"]:
        logger.info("  Weighted Average:")
        for k, v in results["classification_report"]["weighted avg"].items():
            if isinstance(v, (int, float)):
                logger.info(f"    {k}: {v:.4f}")
    
    logger.info("=============================================")


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")


def main():
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting model evaluation")
    
    try:
        # Set default paths if not provided
        if args.model_path is None:
            args.model_path = os.path.join(args.input_dir, "model.joblib")
        
        if args.output_path is None:
            args.output_path = os.path.join(args.input_dir, "evaluation_results.json")
        
        # Load data
        X, y = load_data(args.input_dir, args.input_format)
        logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Check if we have enough data
        if X.shape[0] == 0:
            logger.error("No data loaded; cannot evaluate model.")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model_path)
        
        # Evaluate model
        results = evaluate_model(model, X, y)
        
        feat_csv = os.path.join(args.input_dir, "feature_names.csv")
        try:
            feature_names = pd.read_csv(feat_csv, header=None).iloc[:, 0].tolist()
            importances = model.feature_importances_
            results["feature_importances"] = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        except Exception as e:
            logger.warning(f"Could not load feature importances: {e}")

        # Print results
        print_evaluation_results(results)
        
        # Save results
        save_evaluation_results(results, args.output_path)
        
        logger.info("Model evaluation complete!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())