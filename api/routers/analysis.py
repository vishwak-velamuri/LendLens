# api/routers/analysis.py
import os, uuid, traceback
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from api.schemas import AnalysisResponse
from src.pipeline import run_pipeline    # ← your orchestrator

router = APIRouter()

@router.post(
    "/analyze", 
    response_model=AnalysisResponse,
    summary="Parse PDF, extract metrics, run ML decision"
)
async def analyze(
    file: UploadFile = File(..., description="Bank statement PDF"),
    loan_amount: float   = Form(...),
    down_payment: float  = Form(...),
    interest_rate: float = Form(...),
    term_months: int     = Form(...),
):
    # 1) save upload
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    token   = uuid.uuid4().hex
    path    = os.path.join(tmp_dir, f"{token}_{file.filename}")
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        print(f"File save error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Couldn't save PDF: {e}")

    # 2) call your "combined" pipeline
    try:
        print(f"Processing file: {path}")
        print(f"Loan details: amount={loan_amount}, down={down_payment}, rate={interest_rate}, term={term_months}")
        
        result: AnalysisResponse = run_pipeline(
            pdf_path=path,
            loan_amount=loan_amount,
            down_payment=down_payment,
            interest_rate=interest_rate,
            term_months=term_months,
        )
        
        print(f"Pipeline result: {result}")
        return result
        
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        traceback.print_exc()  # Print stack trace to server console
        raise HTTPException(500, f"Pipeline error: {str(e)}")