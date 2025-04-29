from pydantic import BaseModel
from typing import Dict

class MonthlyTotal(BaseModel):
    deposits: float
    withdrawals: float

class CategoryTotals(BaseModel):
    deposit: float
    withdrawal: float
    withdrawal_regular_bill: float

class Metrics(BaseModel):
    monthly_totals: Dict[str, MonthlyTotal]
    category_totals: CategoryTotals

class AnalysisResponse(BaseModel):
    metrics: Metrics
    decision: str
    confidence: float

    @property
    def monthly_deposits(self) -> Dict[str, float]:
        return {month: data.deposits for month, data in self.metrics.monthly_totals.items()}

    @property
    def monthly_withdrawals(self) -> Dict[str, float]:
        return {month: data.withdrawals for month, data in self.metrics.monthly_totals.items()}

    @property
    def deposit_total(self) -> float:
        return self.metrics.category_totals.deposit

    @property
    def withdrawal_total(self) -> float:
        return self.metrics.category_totals.withdrawal

    @property
    def withdrawal_regular_bill_total(self) -> float:
        return self.metrics.category_totals.withdrawal_regular_bill