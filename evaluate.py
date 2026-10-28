"""
Evaluation and Business Impact Analysis Module
Computes model performance comparisons, confusion matrices, and financial ROI calculations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, recall_score, precision_score, f1_score

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: str = "models/model_metadata.json") -> dict:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run train.py first.")
    with open(metadata_path, "r") as f:
        return json.load(f)


def calculate_business_roi(
    n_at_risk: int,
    avg_monthly_bill: float = 65.0,
    retention_offer_cost: float = 50.0,
    intervention_success_rate: float = 0.35,
    customer_lifespan_months: int = 18
) -> dict:
    """
    Computes business financial return from proactive churn retention intervention.
    
    Parameters:
    - n_at_risk: Number of flagged high-risk customers targeted.
    - avg_monthly_bill: Average revenue per user (ARPU) per month.
    - retention_offer_cost: Cost of retention incentive (e.g., $50 bill credit or promo).
    - intervention_success_rate: Proportion of contacted churning customers saved (typically 30-40%).
    - customer_lifespan_months: Extended customer lifetime if retained.
    """
    total_campaign_cost = n_at_risk * retention_offer_cost
    customers_saved = int(n_at_risk * intervention_success_rate)
    revenue_saved_per_customer = avg_monthly_bill * customer_lifespan_months
    total_revenue_saved = customers_saved * revenue_saved_per_customer
    net_profit_saved = total_revenue_saved - total_campaign_cost
    roi_percentage = (net_profit_saved / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

    return {
        "targeted_customers": n_at_risk,
        "customers_saved": customers_saved,
        "campaign_cost": round(total_campaign_cost, 2),
        "gross_revenue_protected": round(total_revenue_saved, 2),
        "net_business_value": round(net_profit_saved, 2),
        "roi_percentage": round(roi_percentage, 1),
    }


def generate_evaluation_report():
    meta = load_metadata()
    metrics = meta.get("metrics", {})
    xgb_rec = metrics.get("xgboost", {}).get("recall_optimized", {})
    lr_rec = metrics.get("logistic_regression", {}).get("recall_optimized", {})
    
    report_df = pd.DataFrame([
        {
            "Model": "XGBoost (Recall Tuned)",
            "Threshold": xgb_rec.get("threshold"),
            "Recall": f"{xgb_rec.get('recall', 0)*100:.1f}%",
            "Precision": f"{xgb_rec.get('precision', 0)*100:.1f}%",
            "F1-Score": f"{xgb_rec.get('f1', 0):.3f}",
            "ROC-AUC": f"{xgb_rec.get('roc_auc', 0):.3f}",
            "Accuracy": f"{xgb_rec.get('accuracy', 0)*100:.1f}%",
            "False Negatives (Missed Churners)": xgb_rec.get("false_negatives"),
        },
        {
            "Model": "Logistic Regression",
            "Threshold": lr_rec.get("threshold"),
            "Recall": f"{lr_rec.get('recall', 0)*100:.1f}%",
            "Precision": f"{lr_rec.get('precision', 0)*100:.1f}%",
            "F1-Score": f"{lr_rec.get('f1', 0):.3f}",
            "ROC-AUC": f"{lr_rec.get('roc_auc', 0):.3f}",
            "Accuracy": f"{lr_rec.get('accuracy', 0)*100:.1f}%",
            "False Negatives (Missed Churners)": lr_rec.get("false_negatives"),
        }
    ])

    roi_example = calculate_business_roi(n_at_risk=xgb_rec.get("true_positives", 321) + xgb_rec.get("false_positives", 359))

    return report_df, roi_example


if __name__ == "__main__":
    report, roi = generate_evaluation_report()
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(report.to_string(index=False))
    print("\n--- SAMPLE RETENTION ROI SIMULATION ---")
    for k, v in roi.items():
        print(f"  {k}: {v}")
