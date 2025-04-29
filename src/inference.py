import joblib
import logging
from typing import Dict, Any, Optional

# Import feature computation function
from src.feature_computation import compute_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Store loaded model as a module-level singleton
_cached_model = None


def load_model(model_path: str = "data/processed/model.joblib") -> Any:
    global _cached_model
    
    # Return cached model if already loaded
    if _cached_model is not None:
        return _cached_model
        
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        _cached_model = model
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_decision(
    structured: Dict[str, Any],
    form: Dict[str, float],
    model: Optional[Any] = None
) -> Dict[str, Any]:
    # Validate the structured data has required fields
    summary = structured.get("summary", {})
    if "monthly_summaries" not in summary or "overall_summary" not in summary:
        logger.error("Structured data missing required summary fields")
        raise ValueError("Structured data missing required summary fields")
    
    # 1. Load model if not passed in
    if model is None:
        model = load_model()
    
    # 2. Compute features from structured data and form
    feats = compute_features(structured, form)
    
    # 3. Reshape features for model input (from 1D to 2D array)
    feats = feats.reshape(1, -1)

    try:
        confidence = float(model.predict_proba(feats)[0, 1])
        threshold = 0.8
        decision = 1 if confidence >= threshold else 0
    except AttributeError:
        logger.warning("Model doesn't support predict_proba, falling back to .predict()")
        decision = int(model.predict(feats)[0])
        confidence = None
    
    # Map numeric decision to human-readable string
    label_map = {0: "DENY", 1: "APPROVE"}
    decision_str = label_map.get(decision, str(decision))
    
    # 5. Get prediction confidence, handling models that don't support predict_proba
    confidence = None
    try:
        confidence = float(model.predict_proba(feats)[0, 1])
    except AttributeError:
        logger.warning("Model doesn't support predict_proba, confidence not available")
    
    # 6. Return decision, confidence, and financial metrics for UI display
    result = {
        "decision": decision,
        "decision_str": decision_str,
        "metrics": {
            "monthly_summaries": summary["monthly_summaries"],
            "overall_summary": summary["overall_summary"],
            "potential_loans": summary.get("potential_loans", []),
        }
    }
    
    if confidence is not None:
        result["confidence"] = confidence
    
    return result

if __name__ == "__main__":
    import json

    # 1) Load a already‐structured sample statement
    sample = json.load(open("data/structured/IDFC.json", "r"))

    # 2) Create a dummy loan form (values don’t change the banking metrics)
    form = {
        "loan_amount": 25000.0,
        "down_payment": 5000.0,
        "interest_rate": 4.5,
        "term_months": 48
    }

    # 3) Call your inference function
    result = predict_decision(sample, form)

    # 4) Print the full output
    import pprint
    pprint.pprint(result)