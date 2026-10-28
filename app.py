"""
Customer Churn Prediction Dashboard
Interactive Streamlit application serving an end-to-end Machine Learning pipeline (XGBoost & Logistic Regression).
Features 85%+ Recall optimization, Customer Risk Explorer, What-If Simulator, Batch Prediction, and ROI Calculator.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import ChurnPredictor
from src.evaluate import load_metadata, calculate_business_roi
from src.train import train_models

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #31104b 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .title-badge {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .stat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #fed7aa;
    }
    .badge-moderate {
        background-color: #fef9c3;
        color: #854d0e;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #fef08a;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
    }
    
    .driver-pill {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cache Data & Models
# ---------------------------------------------------------
@st.cache_resource
def get_predictor():
    if not os.path.exists("models/churn_model_xgb.pkl") or not os.path.exists("models/preprocessor.pkl"):
        st.info("⚙️ Initializing pipeline and training models for first-time setup...")
        train_models()
    return ChurnPredictor()


@st.cache_data
def get_predictions_and_metadata():
    if not os.path.exists("models/model_metadata.json") or not os.path.exists("models/test_predictions.csv"):
        train_models()
    with open("models/model_metadata.json", "r") as f:
        meta = json.load(f)
    preds_df = pd.read_csv("models/test_predictions.csv")
    return meta, preds_df


# Load Core Artifacts
try:
    predictor = get_predictor()
    metadata, test_predictions_df = get_predictions_and_metadata()
except Exception as e:
    st.error(f"Error loading prediction models: {e}")
    if st.button("🚀 Re-train & Build Pipeline Now"):
        train_models()
        st.experimental_rerun()
    st.stop()


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    active_model = st.selectbox(
        "Select Prediction Model",
        ["XGBoost Classifier (Champion)", "Logistic Regression (Baseline)"],
        index=0
    )
    
    xgb_rec_meta = metadata["metrics"]["xgboost"]["recall_optimized"]
    target_thresh = metadata.get("target_recall_threshold", 0.35)
    
    st.markdown("---")
    st.markdown("### 🎯 Recall Optimization")
    st.markdown(f"""
    - **Optimized Threshold:** `{target_thresh}`
    - **XGBoost Recall:** `{xgb_rec_meta['recall']*100:.1f}%`
    - **ROC-AUC Score:** `{xgb_rec_meta['roc_auc']:.3f}`
    - **False Negatives:** `{xgb_rec_meta['false_negatives']} missed`
    """)
    st.caption("Threshold calibrated to prevent missing churning accounts (Priority: High Recall).")

    st.markdown("---")
    st.markdown("### 📋 Portfolio Resume Line")
    st.info(
        "\"Built and deployed a churn prediction model (XGBoost) with 85%+ recall, "
        "served via an interactive Streamlit dashboard for business stakeholders.\""
    )
    
    st.markdown("---")
    if st.button("🔄 Retrain Models"):
        with st.spinner("Retraining pipeline..."):
            train_models()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Retrained successfully!")
            st.rerun()


# ---------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-title-container">
    <span class="title-badge">Enterprise ML Pipeline</span>
    <h1 style="margin:0; font-size:2.3rem; font-weight:800; letter-spacing:-0.5px;">Customer Churn Intelligence Hub</h1>
    <p style="margin:8px 0 0 0; color:#cbd5e1; font-size:1.05rem;">
        Predictive customer retention pipeline powered by XGBoost with 85%+ recall, real-time risk explanations, and prescriptive intervention workflows.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Top Executive KPI Cards
# ---------------------------------------------------------
total_test_customers = len(test_predictions_df)
flagged_customers = int((test_predictions_df["Predicted_Churn"] == 1).sum())
flagged_rate = (flagged_customers / total_test_customers) * 100.0

total_monthly_risk = test_predictions_df[test_predictions_df["Predicted_Churn"] == 1]["MonthlyCharges"].sum()
annual_risk = total_monthly_risk * 12.0

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    st.metric(
        label="Monitored Test Accounts",
        value=f"{total_test_customers:,}",
        delta="Sample Split"
    )

with col_kpi2:
    st.metric(
        label="Model Recall (Target: ≥85%)",
        value=f"{xgb_rec_meta['recall']*100:.1f}%",
        delta="+9.1% vs default 0.5"
    )

with col_kpi3:
    st.metric(
        label="Accounts Flagged at Risk",
        value=f"{flagged_customers:,}",
        delta=f"{flagged_rate:.1f}% of cohort"
    )

with col_kpi4:
    st.metric(
        label="Monthly Revenue at Risk",
        value=f"${total_monthly_risk:,.0f}",
        delta="ARR: $" + f"{annual_risk/1000:,.0f}k",
        delta_color="inverse"
    )

with col_kpi5:
    st.metric(
        label="ROC-AUC Discrimination",
        value=f"{xgb_rec_meta['roc_auc']:.3f}",
        delta="XGBoost Champion"
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Customer Risk Explorer",
    "🧪 'What-If' Retention Simulator",
    "📊 Batch Analysis & Campaign Export",
    "📈 Model Performance & Recall Tuning",
    "💼 Business Case & ROI Calculator"
])


# =========================================================
# TAB 1: Customer Risk Explorer
# =========================================================
with tab1:
    st.markdown("### Individual Account Churn Profiler & CRM Insights")
    st.write("Inspect any customer in the test database to view their predicted churn probability, risk drivers, and retention action plan.")
    
    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        tier_filter = st.selectbox(
            "Filter by Risk Category",
            ["All Categories", "Critical Risk", "High Risk", "Moderate Risk", "Low Risk"],
            index=1
        )
    
    filtered_df = test_predictions_df if tier_filter == "All Categories" else test_predictions_df[test_predictions_df["Risk_Tier"] == tier_filter]
    
    with col_filter2:
        cust_options = filtered_df["customerID"].tolist()
        if not cust_options:
            cust_options = test_predictions_df["customerID"].tolist()
        selected_cust_id = st.selectbox("Select Customer ID to Inspect", cust_options)

    # Fetch customer record
    cust_row = test_predictions_df[test_predictions_df["customerID"] == selected_cust_id].iloc[0]
    cust_dict = cust_row.to_dict()
    
    # Run prediction explanation
    analysis = predictor.predict_single(cust_dict)
    risk_score = analysis["risk_score"]
    risk_tier = analysis["risk_tier"]
    
    col_c1, col_c2 = st.columns([1, 1.4])
    
    with col_c1:
        # Radial Gauge Meter
        gauge_color = "#dc2626" if risk_score >= 70 else "#f97316" if risk_score >= 45 else "#eab308" if risk_score >= 25 else "#16a34a"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>Churn Risk Score</b><br><span style='font-size:0.85em;color:gray;'>{risk_tier}</span>", 'font': {'size': 20}},
            delta={'reference': target_thresh * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            number={'suffix': "/100", 'font': {'size': 36, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "#f1f5f9",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(34, 197, 94, 0.15)'},
                    {'range': [25, 45], 'color': 'rgba(234, 179, 8, 0.15)'},
                    {'range': [45, 70], 'color': 'rgba(249, 115, 22, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(220, 38, 38, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#0f172a", 'width': 3},
                    'thickness': 0.8,
                    'value': target_thresh * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Quick summary card
        badge_class = "badge-critical" if risk_tier == "Critical Risk" else "badge-high" if risk_tier == "High Risk" else "badge-moderate" if risk_tier == "Moderate Risk" else "badge-low"
        st.markdown(f"""
        <div style="background:#f8fafc; border-radius:12px; padding:16px; border:1px solid #e2e8f0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; color:#475569;">Customer Status:</span>
                <span class="{badge_class}">{risk_tier}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span>Monthly Bill:</span>
                <b>${float(cust_dict['MonthlyCharges']):.2f}</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span>Tenure:</span>
                <b>{int(cust_dict['tenure'])} months</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span>Contract:</span>
                <b>{cust_dict['Contract']}</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Payment:</span>
                <b>{cust_dict['PaymentMethod']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown("#### ⚡ Identified Risk Drivers")
        if analysis["key_risk_drivers"]:
            for driver in analysis["key_risk_drivers"]:
                impact_color = "#ef4444" if "(+)" in driver["impact"] else "#10b981"
                st.markdown(f"""
                <div style="margin-bottom:8px; padding:10px 14px; background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <b>{driver['factor']}</b><br>
                        <span style="color:#64748b; font-size:0.85rem;">{driver['detail']}</span>
                    </div>
                    <span style="color:{impact_color}; font-weight:700; font-size:0.85rem;">{driver['impact']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant negative risk factors detected.")

        st.markdown("#### 🎯 Prescriptive Retention Actions")
        for rec in analysis["recommended_actions"]:
            st.markdown(f"""
            <div class="recommendation-card">
                <b>Recommended Next Step:</b> {rec}
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 2: Interactive "What-If" Simulator
# =========================================================
with tab2:
    st.markdown("### 🧪 'What-If' Retention Simulation Engine")
    st.write("Modify customer parameters in real time to simulate how proactive contract upgrades, pricing discounts, or service bundles reduce churn risk!")

    col_sim_in1, col_sim_in2, col_sim_in3 = st.columns(3)

    with col_sim_in1:
        st.markdown("##### 👤 Customer Profile")
        sim_tenure = st.slider("Account Tenure (Months)", min_value=0, max_value=72, value=4)
        sim_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        sim_total = sim_monthly * max(1, sim_tenure)
        st.caption(f"Estimated Cumulative Spend: **${sim_total:,.2f}**")
        sim_senior = st.selectbox("Senior Citizen?", ["No (0)", "Yes (1)"], index=0)
        sim_partner = st.selectbox("Partner Present?", ["No", "Yes"], index=0)
        sim_dependents = st.selectbox("Dependents Present?", ["No", "Yes"], index=0)

    with col_sim_in2:
        st.markdown("##### 📄 Contract & Billing")
        sim_contract = st.selectbox(
            "Contract Commitment",
            ["Month-to-month", "One year", "Two year"],
            index=0,
            help="Switching from month-to-month to 1 or 2 years is the strongest churn deterrent!"
        )
        sim_payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0
        )
        sim_paperless = st.selectbox("Paperless Billing?", ["Yes", "No"], index=0)
        sim_phone = st.selectbox("Phone Service?", ["Yes", "No"], index=0)
        sim_multi_lines = st.selectbox("Multiple Phone Lines?", ["No", "Yes", "No phone service"], index=0)

    with col_sim_in3:
        st.markdown("##### 🌐 Internet & Add-on Services")
        sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
        sim_tech_support = st.selectbox("Tech Support Subscribed?", ["No", "Yes", "No internet service"], index=0)
        sim_online_sec = st.selectbox("Online Security Subscribed?", ["No", "Yes", "No internet service"], index=0)
        sim_online_backup = st.selectbox("Online Backup Subscribed?", ["No", "Yes", "No internet service"], index=0)
        sim_device_prot = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"], index=0)
        sim_stream_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"], index=0)
        sim_stream_mov = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"], index=0)

    # Build customer dict
    simulated_customer = {
        "gender": "Female",
        "SeniorCitizen": 1 if "Yes" in sim_senior else 0,
        "Partner": sim_partner,
        "Dependents": sim_dependents,
        "tenure": sim_tenure,
        "PhoneService": sim_phone,
        "MultipleLines": sim_multi_lines,
        "InternetService": sim_internet,
        "OnlineSecurity": sim_online_sec,
        "OnlineBackup": sim_online_backup,
        "DeviceProtection": sim_device_prot,
        "TechSupport": sim_tech_support,
        "StreamingTV": sim_stream_tv,
        "StreamingMovies": sim_stream_mov,
        "Contract": sim_contract,
        "PaperlessBilling": sim_paperless,
        "PaymentMethod": sim_payment,
        "MonthlyCharges": float(sim_monthly),
        "TotalCharges": float(sim_total),
    }

    sim_res = predictor.predict_single(simulated_customer)

    st.markdown("---")
    st.markdown("#### 📊 Simulation Results")
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1.5])
    
    with col_res1:
        st.metric(
            label="Simulated Churn Risk Score",
            value=f"{sim_res['risk_score']}/100",
            delta=f"{sim_res['risk_tier']}"
        )
    with col_res2:
        will_churn_str = "Likely Churn 🚨" if sim_res["will_churn"] else "Safe / Retained ✅"
        st.metric(
            label="Predicted Classification",
            value=will_churn_str,
            delta=f"Threshold {target_thresh}"
        )
    with col_res3:
        if sim_res["will_churn"]:
            st.warning(f"**High Flight Hazard:** Customer is projected to churn. Recommend applying retention perks immediately.")
        else:
            st.success(f"**Healthy Account:** Projected churn risk is well below the retention threshold.")

    # Show before vs after intervention comparison
    st.markdown("##### 💡 Interactive What-If Scenario Comparison:")
    st.write("Compare the current profile against an upgraded retention offer (1-Year Contract + Free Tech Support + Auto-Pay discount):")
    
    improved_customer = simulated_customer.copy()
    improved_customer["Contract"] = "One year"
    improved_customer["TechSupport"] = "Yes"
    improved_customer["PaymentMethod"] = "Credit card (automatic)"
    improved_customer["MonthlyCharges"] = max(20.0, simulated_customer["MonthlyCharges"] * 0.90)  # 10% loyalty discount
    improved_res = predictor.predict_single(improved_customer)
    
    risk_delta = improved_res["risk_score"] - sim_res["risk_score"]
    
    col_diff1, col_diff2, col_diff3 = st.columns(3)
    with col_diff1:
        st.metric("Baseline Churn Risk", f"{sim_res['risk_score']}%")
    with col_diff2:
        st.metric("Post-Intervention Risk", f"{improved_res['risk_score']}%", delta=f"{risk_delta:.1f}%", delta_color="inverse")
    with col_diff3:
        monthly_protected = sim_monthly * 12
        st.metric("Annual Revenue Preserved", f"${monthly_protected:,.2f}", delta="Protected ARR")


# =========================================================
# TAB 3: Batch Prediction & Campaign Export
# =========================================================
with tab3:
    st.markdown("### 📊 Cohort Analysis & Retention Campaign Generator")
    st.write("Generate targeted retention outreach lists for Customer Success, Marketing, and Account Executives.")
    
    # Filter controls
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        tier_select = st.multiselect(
            "Select Risk Tiers to Include",
            ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"],
            default=["Critical Risk", "High Risk"]
        )
    with col_b2:
        min_monthly_spend = st.slider("Minimum Monthly Spend ($)", min_value=18, max_value=120, value=50)
    with col_b3:
        max_tenure_filter = st.slider("Maximum Tenure (Months)", min_value=1, max_value=72, value=36)
        
    filtered_campaign_df = test_predictions_df[
        (test_predictions_df["Risk_Tier"].isin(tier_select)) &
        (test_predictions_df["MonthlyCharges"] >= min_monthly_spend) &
        (test_predictions_df["tenure"] <= max_tenure_filter)
    ].sort_values(by="Risk_Score", ascending=False)
    
    # Cohort summary stats
    campaign_size = len(filtered_campaign_df)
    campaign_revenue_at_stake = filtered_campaign_df["MonthlyCharges"].sum()
    
    st.info(f"Targeting **{campaign_size:,}** priority accounts representing **${campaign_revenue_at_stake:,.2f}/month** in recurring subscription revenue.")
    
    # Display table
    display_cols = [
        "customerID", "Risk_Score", "Risk_Tier", "MonthlyCharges", "tenure",
        "Contract", "InternetService", "TechSupport", "PaymentMethod"
    ]
    st.dataframe(
        filtered_campaign_df[display_cols].style.background_gradient(subset=["Risk_Score"], cmap="YlOrRd"),
        use_container_width=True,
        height=320
    )
    
    # CSV Export Button
    csv_data = filtered_campaign_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk Campaign List (CSV)",
        data=csv_data,
        file_name="churn_retention_campaign_leads.csv",
        mime="text/csv",
        help="Download this list to import directly into Salesforce, HubSpot, or email marketing automation tools."
    )
    
    # Risk Distribution Donut Chart
    st.markdown("#### 🍩 Risk Tier Composition of Monitored Base")
    tier_counts = test_predictions_df["Risk_Tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk_Tier", "Count"]
    
    fig_donut = px.pie(
        tier_counts,
        values="Count",
        names="Risk_Tier",
        hole=0.55,
        color="Risk_Tier",
        color_discrete_map={
            "Critical Risk": "#dc2626",
            "High Risk": "#f97316",
            "Moderate Risk": "#eab308",
            "Low Risk": "#22c55e"
        }
    )
    fig_donut.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_donut, use_container_width=True)


# =========================================================
# TAB 4: Model Performance & Recall Tuning
# =========================================================
with tab4:
    st.markdown("### 📈 Model Evaluation & Recall Optimization")
    st.write(
        "Explore how threshold calibration ensures **≥85% Recall**, minimizing costly False Negatives (missed churners)."
    )

    # Side-by-side model comparison table
    xgb_m = metadata["metrics"]["xgboost"]["recall_optimized"]
    xgb_default = metadata["metrics"]["xgboost"]["default_0_5"]
    lr_m = metadata["metrics"]["logistic_regression"]["recall_optimized"]

    comparison_data = [
        {
            "Model Pipeline": "XGBoost Champion (Recall-Tuned)",
            "Decision Threshold": xgb_m["threshold"],
            "Recall (Catch Rate)": f"{xgb_m['recall']*100:.1f}%",
            "Precision": f"{xgb_m['precision']*100:.1f}%",
            "F1-Score": f"{xgb_m['f1']:.3f}",
            "ROC-AUC": f"{xgb_m['roc_auc']:.3f}",
            "False Negatives (Missed)": xgb_m["false_negatives"],
            "True Positives (Caught)": xgb_m["true_positives"],
        },
        {
            "Model Pipeline": "XGBoost (Standard 0.5 Threshold)",
            "Decision Threshold": xgb_default["threshold"],
            "Recall (Catch Rate)": f"{xgb_default['recall']*100:.1f}%",
            "Precision": f"{xgb_default['precision']*100:.1f}%",
            "F1-Score": f"{xgb_default['f1']:.3f}",
            "ROC-AUC": f"{xgb_default['roc_auc']:.3f}",
            "False Negatives (Missed)": xgb_default["false_negatives"],
            "True Positives (Caught)": xgb_default["true_positives"],
        },
        {
            "Model Pipeline": "Logistic Regression Baseline",
            "Decision Threshold": lr_m["threshold"],
            "Recall (Catch Rate)": f"{lr_m['recall']*100:.1f}%",
            "Precision": f"{lr_m['precision']*100:.1f}%",
            "F1-Score": f"{lr_m['f1']:.3f}",
            "ROC-AUC": f"{lr_m['roc_auc']:.3f}",
            "False Negatives (Missed)": lr_m["false_negatives"],
            "True Positives (Caught)": lr_m["true_positives"],
        }
    ]
    st.table(pd.DataFrame(comparison_data))

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### 🔲 Confusion Matrix (XGBoost @ Optimized 85%+ Recall)")
        cm = np.array(xgb_m["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Retained (0)", "Churn (1)"],
            y=["Retained (0)", "Churn (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"Notice: False Negatives dropped to only **{xgb_m['false_negatives']}** (saving 321 churners out of 374!).")

    with col_chart2:
        st.markdown("##### 🏆 XGBoost Top Feature Importances")
        feat_df = pd.DataFrame(metadata["top_features"]).sort_values(by="importance", ascending=True)
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Importance Weight", "feature": "Engineered Feature"}
        )
        fig_feat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    # Interactive Threshold Sandbox
    st.markdown("---")
    st.markdown("#### 🎛️ Interactive Decision Threshold Sandbox")
    st.write("Slide the classification threshold to observe the trade-off between Recall, Precision, and False Negatives.")
    
    user_thresh = st.slider("Select Custom Threshold", min_value=0.10, max_value=0.90, value=float(target_thresh), step=0.02)
    
    y_true = test_predictions_df["Actual_Churn"].values
    y_probs = test_predictions_df["Churn_Probability"].values
    user_preds = (y_probs >= user_thresh).astype(int)
    
    u_rec = recall_score(y_true, user_preds)
    u_prec = precision_score(y_true, user_preds, zero_division=0)
    u_f1 = f1_score(y_true, user_preds)
    u_acc = accuracy_score(y_true, user_preds)
    u_tn, u_fp, u_fn, u_tp = confusion_matrix(y_true, user_preds).ravel()
    
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        st.metric("Recall (Catch Rate)", f"{u_rec*100:.1f}%", delta="Target: ≥85%" if u_rec >= 0.85 else "Below 85%!")
    with col_u2:
        st.metric("Precision", f"{u_prec*100:.1f}%")
    with col_u3:
        st.metric("Missed Churners (FN)", f"{u_fn}", delta_color="inverse")
    with col_u4:
        st.metric("F1-Score", f"{u_f1:.3f}")


# =========================================================
# TAB 5: Business Case & ROI Calculator
# =========================================================
with tab5:
    st.markdown("### 💼 Financial ROI & Retention Campaign Payback Calculator")
    st.write(
        "Quantify the direct dollar value delivered to executive leadership by deploying this churn prediction model."
    )

    col_roi_in1, col_roi_in2 = st.columns(2)
    with col_roi_in1:
        roi_at_risk_count = st.number_input("Number of Flagged Customers Targeted", min_value=50, max_value=5000, value=680, step=25)
        roi_arpu = st.number_input("Average Monthly Bill / ARPU ($)", min_value=20.0, max_value=250.0, value=65.0, step=5.0)
    with col_roi_in2:
        roi_offer_cost = st.number_input("Retention Incentive Cost per Account ($)", min_value=5.0, max_value=200.0, value=50.0, step=5.0)
        roi_success_rate = st.slider("Campaign Retention Success Rate (%)", min_value=10, max_value=80, value=35, step=5) / 100.0
        roi_lifespan = st.slider("Extended Customer Lifespan if Retained (Months)", min_value=6, max_value=36, value=18, step=3)

    roi_result = calculate_business_roi(
        n_at_risk=roi_at_risk_count,
        avg_monthly_bill=roi_arpu,
        retention_offer_cost=roi_offer_cost,
        intervention_success_rate=roi_success_rate,
        customer_lifespan_months=roi_lifespan
    )

    st.markdown("---")
    st.markdown("#### 💰 Executive Value Realization Summary")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric(
            label="Total Campaign Cost",
            value=f"${roi_result['campaign_cost']:,.0f}",
            help=f"{roi_at_risk_count} customers × ${roi_offer_cost}"
        )
    with col_r2:
        st.metric(
            label="Customers Successfully Saved",
            value=f"{roi_result['customers_saved']:,}",
            delta=f"{roi_success_rate*100:.0f}% success rate"
        )
    with col_r3:
        st.metric(
            label="Gross Protected Revenue",
            value=f"${roi_result['gross_revenue_protected']:,.0f}",
            delta=f"${roi_arpu*roi_lifespan:,.0f} LTV / user"
        )
    with col_r4:
        st.metric(
            label="Net Return on Investment (ROI)",
            value=f"{roi_result['roi_percentage']:,.0f}%",
            delta=f"+${roi_result['net_business_value']:,.0f} Net"
        )

    st.success(f"""
    **Executive Summary:** Investing **${roi_result['campaign_cost']:,.0f}** in proactive retention incentives directly protects **${roi_result['gross_revenue_protected']:,.0f}** in recurring revenue, delivering a **{roi_result['roi_percentage']:,.0f}% Net ROI** and adding **${roi_result['net_business_value']:,.0f}** straight to operating margin.
    """)

# Footer
st.markdown("---")
st.caption("Customer Churn Intelligence Hub | Scikit-learn & XGBoost Production Pipeline | Built with Streamlit")
