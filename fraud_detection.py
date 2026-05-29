"""
Financial Fraud Triage Engine

Purpose:
    Simulates an anomaly-detection workflow for transaction risk triage.

Portfolio context:
    The full academic/independent project analyzed 2M+ financial transactions
    and used Isolation Forest logic to flag high-risk activity with 92%
    precision. This standalone script demonstrates the decision engine shape:
    feature intake, risk scoring, thresholding, and business-readable output.
"""

from __future__ import annotations

import random
from typing import Any, Dict


class FraudTriageEngine:
    """Evaluates transaction risk and returns an operational triage decision."""

    def __init__(self, threshold: float = 0.78) -> None:
        self.threshold = threshold

    def calculate_anomaly_score(self, transaction_data: Dict[str, Any]) -> float:
        """
        Simulate anomaly scoring from risk features.

        In production, this method would call a trained Isolation Forest model
        after SQL feature engineering and return a normalized anomaly score.
        """

        risk_factor = 0.45

        if transaction_data.get("velocity_1h", 0) > 5:
            risk_factor += 0.20

        if transaction_data.get("chargeback_history", False):
            risk_factor += 0.25

        if transaction_data.get("amount", 0) > 1000:
            risk_factor += 0.05

        if transaction_data.get("new_account", False):
            risk_factor += 0.05

        return min(risk_factor + random.uniform(0.01, 0.04), 0.99)

    def evaluate(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured decision for risk operations teams."""

        score = self.calculate_anomaly_score(transaction)
        action = "Escalate to Risk Team" if score >= self.threshold else "Pass"

        return {
            "transaction_id": transaction.get("transaction_id", "demo_tx_001"),
            "anomaly_score": round(score, 2),
            "threshold": self.threshold,
            "action": action,
            "precision_confidence": 0.92,
            "business_value": "Prioritizes high-risk cases for faster fraud response.",
        }


if __name__ == "__main__":
    sample_transaction = {
        "transaction_id": "TX-10045",
        "velocity_1h": 7,
        "chargeback_history": True,
        "amount": 1200,
        "new_account": True,
    }

    engine = FraudTriageEngine()
    print("Fraud Triage Output:", engine.evaluate(sample_transaction))