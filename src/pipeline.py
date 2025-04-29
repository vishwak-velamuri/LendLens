import os
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

from api.schemas import AnalysisResponse, Metrics, CategoryTotals, MonthlyTotal
from src.structurer import analyze_statement
from src.feature_computation import compute_features
from src.inference import predict_decision

def run_pipeline(
    pdf_path: str,
    loan_amount: float,
    down_payment: float,
    interest_rate: float,
    term_months: int,
) -> AnalysisResponse:
    try:
        logger.info(f"Starting pipeline with pdf: {pdf_path}")
        logger.info(f"Loan parameters: amount={loan_amount}, down={down_payment}, rate={interest_rate}, term={term_months}")

        # infer bank_name from filename (so structurer can pick the right schema)
        basename = os.path.basename(pdf_path)
        if "_" in basename:
            # your saved files look like: <uuid>_<OriginalName>.pdf
            _, original = basename.split("_", 1)
            bank_name = os.path.splitext(original)[0]
        else:
            bank_name = os.path.splitext(basename)[0]
        
        logger.info(f"Detected bank name: {bank_name}")

        # step 1: full LLM + PDF → { "transactions":…, "summary":… }
        logger.info("Starting statement analysis...")
        structured = analyze_statement(pdf_path, bank_name)
        logger.info("Statement analysis complete")

        # step 2: build the feature vector for XGBoost
        form = {
            "loan_amount": loan_amount,
            "down_payment": down_payment,
            "interest_rate": interest_rate,
            "term_months": term_months,
        }
        logger.info("Computing features...")
        feature_vector = compute_features(structured, form)
        logger.info("Feature computation complete")

        # step 3: run your model to get back decision, confidence, and raw metrics
        logger.info("Running prediction model...")
        pred = predict_decision(structured, form)
        decision_str = pred["decision_str"]
        confidence = pred.get("confidence", 0.0)
        metrics = pred["metrics"]
        logger.info(f"Prediction complete: {decision_str} with confidence {confidence}")

        # step 4: format data for frontend
        logger.info("Formatting response...")
        
        # Create monthly_totals in the format the frontend expects
        monthly_totals = {}
        for m in metrics["monthly_summaries"]:
            month = m["month"]
            monthly_totals[month] = MonthlyTotal(
                deposits=m["total_deposits"],
                withdrawals=m["total_withdrawals"]
            )

        # Create category_totals in the format the frontend expects
        overall_cats = metrics["overall_summary"].get("categories", {})
        category_totals = CategoryTotals(
            deposit=overall_cats.get("deposit", 0.0),
            withdrawal=overall_cats.get("withdrawal", 0.0),
            withdrawal_regular_bill=overall_cats.get("withdrawal_regular_bill", 0.0)
        )

        # Build the final response
        response = AnalysisResponse(
            metrics=Metrics(
                monthly_totals=monthly_totals,
                category_totals=category_totals
            ),
            decision=decision_str,
            confidence=confidence
        )
        
        logger.info("Pipeline complete, returning response")
        return response
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.error(traceback.format_exc())
        raise