"""
Inference & Risk Explanation Engine
Provides single-customer and batch prediction capabilities, risk score categorization,
and rule-based feature attribution for actionable retention recommendations.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.feature_engineering import load_preprocessor
except ImportError:
    from feature_engineering import load_preprocessor

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.35  # Tuned for >= 85% recall


class ChurnPredictor:
    def __init__(
        self,
        model_path: str = "models/churn_model_xgb.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
        metadata_path: str = "models/model_metadata.json"
    ):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Trained model or preprocessor not found. Run src/train.py first.")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        self.threshold = DEFAULT_THRESHOLD
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    self.threshold = meta.get("target_recall_threshold", DEFAULT_THRESHOLD)
            except Exception:
                pass

    def _determine_risk_tier(self, prob: float) -> str:
        if prob >= 0.70:
            return "Critical Risk"
        elif prob >= 0.45:
            return "High Risk"
        elif prob >= 0.25:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def _generate_retention_recommendations(self, customer: dict, prob: float) -> list:
        recommendations = []
        contract = customer.get("Contract", "")
        tech_support = customer.get("TechSupport", "")
        online_sec = customer.get("OnlineSecurity", "")
        payment = customer.get("PaymentMethod", "")
        monthly = float(customer.get("MonthlyCharges", 0.0))
        tenure = int(customer.get("tenure", 0))

        if prob >= 0.35:
            if contract == "Month-to-month":
                recommendations.append("Offer 15% discount incentive to upgrade to a 1-Year or 2-Year contract.")
            if tech_support != "Yes":
                recommendations.append("Provide complimentary 3-month Tech Support upgrade to address service friction.")
            if online_sec != "Yes":
                recommendations.append("Bundle complimentary Online Security & Device Protection add-on.")
            if "Electronic check" in payment:
                recommendations.append("Incentivize switch to Auto-Pay with a one-time $10 account credit.")
            if monthly > 75.0 and tenure < 12:
                recommendations.append("Review billing plan: offer mid-tier loyalty plan with reduced monthly rate.")
        
        if not recommendations:
            recommendations.append("Customer is healthy. Maintain standard customer success check-ins and quarterly review.")

        return recommendations

    def _identify_key_risk_drivers(self, customer: dict) -> list:
        drivers = []
        contract = customer.get("Contract", "")
        tenure = int(customer.get("tenure", 0))
        monthly = float(customer.get("MonthlyCharges", 0.0))
        payment = customer.get("PaymentMethod", "")
        internet = customer.get("InternetService", "")
        tech_support = customer.get("TechSupport", "")
        online_sec = customer.get("OnlineSecurity", "")

        if contract == "Month-to-month":
            drivers.append({"factor": "Month-to-Month Contract", "impact": "High Risk Driver (+)", "detail": "Short-term commitment increases flight risk"})
        elif contract in ["One year", "Two year"]:
            drivers.append({"factor": f"{contract} Contract", "impact": "Protective Factor (-)", "detail": "Long-term agreement provides stability"})

        if tenure <= 6:
            drivers.append({"factor": "New Customer (Tenure <= 6 mos)", "impact": "High Risk Driver (+)", "detail": "Highest churn hazard occurs in first 6 months"})
        elif tenure >= 36:
            drivers.append({"factor": "High Tenure (> 3 years)", "impact": "Protective Factor (-)", "detail": "Strong brand loyalty and habituation"})

        if internet == "Fiber optic" and monthly > 80:
            drivers.append({"factor": "Premium Fiber Optic Spend", "impact": "Moderate Risk Driver (+)", "detail": "High price sensitivity compared to competition"})

        if tech_support == "No":
            drivers.append({"factor": "No Tech Support", "impact": "Moderate Risk Driver (+)", "detail": "Unresolved technical issues trigger churn"})

        if online_sec == "No":
            drivers.append({"factor": "No Online Security", "impact": "Risk Driver (+)", "detail": "Lower service stickiness and switching cost"})

        if "Electronic check" in payment:
            drivers.append({"factor": "Manual Electronic Check", "impact": "Moderate Risk Driver (+)", "detail": "Non-automatic payments have highest default/churn rates"})

        return drivers

    def predict_single(self, customer_data: dict) -> dict:
        """
        Runs prediction and risk assessment for a single customer dictionary.
        """
        df = pd.DataFrame([customer_data])
        X_trans = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        risk_score = round(prob * 100.0, 1)
        risk_tier = self._determine_risk_tier(prob)
        predicted_churn = int(prob >= self.threshold)

        drivers = self._identify_key_risk_drivers(customer_data)
        recommendations = self._generate_retention_recommendations(customer_data, prob)

        return {
            "churn_probability": round(prob, 4),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "decision_threshold": self.threshold,
            "will_churn": bool(predicted_churn),
            "key_risk_drivers": drivers,
            "recommended_actions": recommendations,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch predictions over a DataFrame.
        Returns a sorted DataFrame with probabilities, scores, and tiers.
        """
        df_copy = df.copy()
        X_trans = self.preprocessor.transform(df_copy)
        probs = self.model.predict_proba(X_trans)[:, 1]
        
        df_copy["Churn_Probability"] = np.round(probs, 4)
        df_copy["Risk_Score"] = np.round(probs * 100.0, 1)
        df_copy["Predicted_Churn"] = (probs >= self.threshold).astype(int)
        df_copy["Risk_Tier"] = [self._determine_risk_tier(p) for p in probs]
        
        # Sort descending by risk score
        return df_copy.sort_values(by="Risk_Score", ascending=False)


if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 139.70,
    }
    result = predictor.predict_single(sample_customer)
    print("\n--- SINGLE CUSTOMER INFERENCE RESULT ---")
    print(f"Risk Score: {result['risk_score']} / 100 ({result['risk_tier']})")
    print(f"Predicted to Churn: {result['will_churn']} (Threshold: {result['decision_threshold']})")
    print("Risk Drivers:")
    for d in result["key_risk_drivers"]:
        print(f"  * {d['factor']} [{d['impact']}]: {d['detail']}")
    print("Recommendations:")
    for r in result["recommended_actions"]:
        print(f"  -> {r}")
