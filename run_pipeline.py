"""
Customer Churn Prediction - End-to-End Pipeline Runner
Runs data acquisition, preprocessing, model training, threshold tuning (for 85%+ recall),
and generates all artifacts required for the Streamlit dashboard.
"""

import sys
import os

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train import train_models
from src.evaluate import generate_evaluation_report

def main():
    print("=" * 65)
    print("  CUSTOMER CHURN PREDICTION PIPELINE - EXECUTION START")
    print("=" * 65)
    
    # 1. Train models and generate metadata
    metadata = train_models()
    
    # 2. Generate summary report and ROI analysis
    report_df, roi = generate_evaluation_report()
    
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON MATRIX")
    print("=" * 65)
    print(report_df.to_string(index=False))
    
    print("\n" + "=" * 65)
    print("  ESTIMATED RETENTION CAMPAIGN ROI")
    print("=" * 65)
    print(f"  Targeted High-Risk Customers: {roi['targeted_customers']}")
    print(f"  Protected Gross Revenue:     ${roi['gross_revenue_protected']:,.2f}")
    print(f"  Net Value Created:           ${roi['net_business_value']:,.2f}")
    print(f"  Campaign ROI:                 {roi['roi_percentage']}%")
    print("=" * 65)
    print("\nPipeline finished successfully! Launch the dashboard with:")
    print("  streamlit run app.py\n")

if __name__ == "__main__":
    main()
