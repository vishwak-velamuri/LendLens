#!/usr/bin/env python3
"""
XGBoost Model Training Script

This script loads the preprocessed features, trains an XGBoost classifier using 
cross-validation, and saves the final model along with performance metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
import logging
import argparse
import joblib
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any

from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

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
    parser = argparse.ArgumentParser(description="Train XGBoost model on financial features")
    parser.add_argument("--input-dir", default="data/processed",
                        help="Directory containing processed features")
    parser.add_argument("--config-path", default="src/xgb_params.yaml",
                        help="Path to XGBoost parameters YAML file")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directory to save trained model and metadata")
    parser.add_argument("--cv-folds", type=int, default=None,
                        help="Number of CV folds (default: leave-one-out for small datasets)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--metric", default="accuracy", choices=["auc", "accuracy"],
                        help="Evaluation metric for CV")
    parser.add_argument("--input-format", default="auto", choices=["auto", "numpy", "csv"],
                        help="Format of input data (auto will try numpy first, then csv)")
    return parser.parse_args()


def load_data(input_dir: str, input_format: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from processed data.
    
    Args:
        input_dir: Directory containing processed data
        input_format: Format of input data ("auto", "numpy", or "csv")
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
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


def load_hyperparameters(config_path: str) -> Dict[str, Any]:
    """
    Load XGBoost hyperparameters from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary of hyperparameters
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded hyperparameters from {config_path}")
        return params
    except Exception as e:
        logger.warning(f"Failed to load hyperparameters from {config_path}: {e}")
        logger.info("Using default hyperparameters")
        
        # Default parameters if config file is not found
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1
        }


def setup_cross_validation(n_samples: int, n_folds: Optional[int], random_seed: int) -> Any:
    """
    Set up cross-validation strategy based on dataset size.
    
    Args:
        n_samples: Number of samples in the dataset
        n_folds: Number of CV folds (or None for auto)
        random_seed: Random seed for reproducibility
        
    Returns:
        Cross-validation iterator
    """
    if n_folds is None:
        # Auto-select CV strategy based on dataset size
        if n_samples < 10:
            logger.info(f"Small dataset detected ({n_samples} samples). Using Leave-One-Out CV.")
            return LeaveOneOut()
        else:
            n_folds = min(5, n_samples)
            logger.info(f"Using {n_folds}-fold cross-validation")
            return KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    else:
        # Use specified number of folds
        if n_folds >= n_samples:
            logger.warning(f"Requested {n_folds} folds but only have {n_samples} samples. Using Leave-One-Out CV.")
            return LeaveOneOut()
        else:
            logger.info(f"Using {n_folds}-fold cross-validation as specified")
            return KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)


def run_cross_validation(
    X: np.ndarray, 
    y: np.ndarray, 
    params: Dict[str, Any], 
    cv: Any, 
    metric: str, 
    random_seed: int
) -> Dict[str, float]:
    """
    Run cross-validation and return performance metrics.
    
    Args:
        X: Feature matrix
        y: Target vector
        params: XGBoost hyperparameters
        cv: Cross-validation iterator
        metric: Evaluation metric ('auc' or 'accuracy')
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with CV performance metrics
    """
    # Clone params to avoid modifying the original
    model_params = params.copy()
    
    # Extract n_estimators if present (not an XGBoost parameter but a classifier init param)
    n_estimators = model_params.pop('n_estimators', 100)
    n_jobs = model_params.pop('n_jobs', None)
    
    if 'scale_pos_weight' not in model_params:
        class_ratio = np.sum(y == 0) / np.sum(y == 1)
        model_params['scale_pos_weight'] = class_ratio
        logger.info(f"Auto-computed scale_pos_weight: {class_ratio:.2f}")

    # Create classifier with specified parameters
    clf = xgb.XGBClassifier(
        **model_params,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_seed
    )
    
    # Select scoring metric
    scoring = 'roc_auc' if metric == 'auc' else 'accuracy'
    
    # Run cross-validation
    logger.info(f"Running cross-validation with metric: {scoring}")
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    
    # Calculate performance statistics
    results = {
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
        'scores': scores.tolist(),
        'metric': metric
    }
    
    logger.info(f"CV {metric} score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    logger.info(f"Min score: {results['min_score']:.4f}, Max score: {results['max_score']:.4f}")
    
    return results


def train_final_model(
    X: np.ndarray, 
    y: np.ndarray, 
    params: Dict[str, Any], 
    random_seed: int
) -> xgb.XGBClassifier:
    """
    Train the final model on the full dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        params: XGBoost hyperparameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained XGBoost classifier
    """
    # Clone params to avoid modifying the original
    model_params = params.copy()
    
    # Extract n_estimators if present
    n_estimators = model_params.pop('n_estimators', 100)
    n_jobs = model_params.pop('n_jobs', None)
    
    # Create and train the model
    logger.info("Training final model on full dataset")
    model = xgb.XGBClassifier(
        **model_params,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_seed
    )
    
    model.fit(X, y)
    logger.info("Final model training complete")
    
    return model


def save_model_and_metadata(
    model: xgb.XGBClassifier,
    params: Dict[str, Any],
    cv_results: Dict[str, float],
    output_dir: str,
    X: np.ndarray,
    random_seed: int
) -> None:
    """
    Save the trained model and metadata.
    
    Args:
        model: Trained XGBoost classifier
        params: Hyperparameters used for training
        cv_results: Cross-validation results
        output_dir: Directory to save model and metadata
        X: Feature matrix (for extracting feature importance)
        random_seed: Random seed used for training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Create metadata
    metadata = {
        "model_type": "XGBoost Classifier",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": params,
        "cross_validation": cv_results,
        "dataset_size": {
            "samples": X.shape[0],
            "features": X.shape[1]
        },
        "random_seed": random_seed
    }
    
    # Get feature importance if possible
    try:
        importance = model.feature_importances_
        feature_names_path = os.path.join(output_dir, "feature_names.csv")
        
        if os.path.exists(feature_names_path):
            feature_names = pd.read_csv(feature_names_path, header=None).iloc[:, 0].tolist()
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            metadata["feature_importance"] = dict(sorted_importance)
        else:
            metadata["feature_importance"] = {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")


def main():
    """Main function to run the model training pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting model training")
    
    # Load data
    X, y = load_data(args.input_dir, args.input_format)
    logger.info(f"Loading from: {args.input_dir}/X.npy, y.npy")
    logger.info(f"X shape: {X.shape}, unique y: {np.unique(y, return_counts=True)}")
    logger.info(f"First row of X: {X[0]}")
    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Check if we have enough data
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable has only one class. Cannot train a classifier.")
    
    # Load hyperparameters
    params = load_hyperparameters(args.config_path)
    logger.info(f"Hyperparameters: {params}")
    
    # Setup cross-validation
    cv = setup_cross_validation(X.shape[0], args.cv_folds, args.random_seed)
    
    # Run cross-validation
    cv_results = run_cross_validation(X, y, params, cv, args.metric, args.random_seed)
    
    # Train final model
    final_model = train_final_model(X, y, params, args.random_seed)
    
    # Save model and metadata
    save_model_and_metadata(final_model, params, cv_results, args.output_dir, X, args.random_seed)
    
    logger.info("Model training complete!")


if __name__ == "__main__":
    main()