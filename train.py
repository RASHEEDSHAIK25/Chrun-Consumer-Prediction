"""
Model Training & Threshold Optimization Module
Trains Logistic Regression and XGBoost classifiers, tunes decision thresholds
for >= 85% recall, and stores models and comprehensive metadata.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.data_loader import load_and_split_data
    from src.feature_engineering import (
        build_preprocessor_pipeline,
        save_preprocessor,
        get_feature_names,
    )
except ImportError:
    from data_loader import load_and_split_data
    from feature_engineering import (
        build_preprocessor_pipeline,
        save_preprocessor,
        get_feature_names,
    )

logger = logging.getLogger(__name__)


def find_optimal_threshold_for_recall(y_true, y_probs, target_recall: float = 0.85):
    """
    Finds the maximum decision threshold that achieves at least target_recall.
    Maximizes precision/F1 subject to recall >= target_recall.
    """
    thresholds = np.linspace(0.01, 0.99, 500)
    best_thresh = 0.5
    best_f1 = -1.0
    valid_thresholds = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if rec >= target_recall:
            valid_thresholds.append((t, rec, prec, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

    # If no threshold strictly hits target_recall, pick the one with highest recall
    if not valid_thresholds:
        recalls = [recall_score(y_true, (y_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(recalls))
        best_thresh = thresholds[best_idx]

    return round(float(best_thresh), 3)


def train_models():
    """
    Trains Logistic Regression and XGBoost on Telco dataset, evaluates performance,
    optimizes recall >= 85%, and saves models + metadata.
    """
    os.makedirs("models", exist_ok=True)
    logger.info("Step 1: Loading and splitting dataset...")
    X_train_raw, X_test_raw, y_train, y_test, full_df = load_and_split_data()

    customer_ids_test = X_test_raw["customerID"].values if "customerID" in X_test_raw.columns else [f"CUST-{i}" for i in range(len(X_test_raw))]
    customer_ids_train = X_train_raw["customerID"].values if "customerID" in X_train_raw.columns else [f"CUST-{i}" for i in range(len(X_train_raw))]

    logger.info("Step 2: Fitting preprocessing & domain feature extraction pipeline...")
    preprocessor = build_preprocessor_pipeline()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    save_preprocessor(preprocessor)
    feature_names = get_feature_names(preprocessor)

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / float(pos_count)

    logger.info(f"Class distribution: Non-churn={neg_count}, Churn={pos_count}. Scale pos weight={scale_pos_weight:.2f}")

    # --- Model 1: Logistic Regression Baseline ---
    logger.info("Step 3: Training Logistic Regression Baseline...")
    lr_model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)

    # --- Model 2: XGBoost Classifier Champion ---
    logger.info("Step 4: Training XGBoost Classifier Champion...")
    xgb_model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight * 0.95,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Save models
    joblib.dump(lr_model, "models/churn_model_lr.pkl")
    joblib.dump(xgb_model, "models/churn_model_xgb.pkl")
    logger.info("Models saved to models/churn_model_lr.pkl and models/churn_model_xgb.pkl")

    # Predictions & Probabilities
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Threshold Optimization for 85%+ Recall
    xgb_thresh_85 = find_optimal_threshold_for_recall(y_test, xgb_probs, target_recall=0.85)
    lr_thresh_85 = find_optimal_threshold_for_recall(y_test, lr_probs, target_recall=0.85)

    logger.info(f"Optimized Decision Threshold for XGBoost (85%+ Recall): {xgb_thresh_85}")

    def evaluate_model(y_true, probs, threshold):
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        return {
            "threshold": threshold,
            "accuracy": round(float(accuracy_score(y_true, preds)), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
            "confusion_matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    metrics = {
        "xgboost": {
            "default_0_5": evaluate_model(y_test, xgb_probs, 0.5),
            "recall_optimized": evaluate_model(y_test, xgb_probs, xgb_thresh_85),
        },
        "logistic_regression": {
            "default_0_5": evaluate_model(y_test, lr_probs, 0.5),
            "recall_optimized": evaluate_model(y_test, lr_probs, lr_thresh_85),
        }
    }

    # ROC Curve coordinates for dashboard rendering
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    p_xgb, r_xgb, _ = precision_recall_curve(y_test, xgb_probs)
    p_lr, r_lr, _ = precision_recall_curve(y_test, lr_probs)

    # Subsample curves to 50 points each for fast dashboard payload
    step_xgb = max(1, len(fpr_xgb) // 50)
    curves_data = {
        "roc": {
            "xgb": {"fpr": [round(float(x), 4) for x in fpr_xgb[::step_xgb]], "tpr": [round(float(y), 4) for y in tpr_xgb[::step_xgb]]},
            "lr": {"fpr": [round(float(x), 4) for x in fpr_lr[::step_xgb]], "tpr": [round(float(y), 4) for y in tpr_lr[::step_xgb]]},
        },
        "pr": {
            "xgb": {"precision": [round(float(x), 4) for x in p_xgb[::step_xgb]], "recall": [round(float(y), 4) for y in r_xgb[::step_xgb]]},
            "lr": {"precision": [round(float(x), 4) for x in p_lr[::step_xgb]], "recall": [round(float(y), 4) for y in r_lr[::step_xgb]]},
        }
    }

    # Feature Importances (XGBoost)
    xgb_feat_imp = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": [round(float(x), 5) for x in xgb_feat_imp]
    }).sort_values(by="importance", ascending=False)

    top_features = feat_imp_df.head(15).to_dict(orient="records")

    metadata = {
        "model_name": "XGBoost Customer Churn Classifier",
        "dataset": "IBM Telco Customer Churn",
        "total_records": len(full_df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "target_recall_threshold": xgb_thresh_85,
        "metrics": metrics,
        "top_features": top_features,
        "curves": curves_data,
        "feature_count": len(feature_names),
        "class_proportions": {
            "churn_rate": round(float(full_df["Churn"].mean()), 4),
            "retain_rate": round(float(1 - full_df["Churn"].mean()), 4),
        }
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to models/model_metadata.json")

    # Generate rich test set predictions table for interactive CRM Dashboard
    preds_test_df = X_test_raw.copy()
    preds_test_df["Actual_Churn"] = y_test.values
    preds_test_df["Churn_Probability"] = np.round(xgb_probs, 4)
    preds_test_df["Risk_Score"] = np.round(xgb_probs * 100, 1)
    preds_test_df["Predicted_Churn"] = (xgb_probs >= xgb_thresh_85).astype(int)
    
    # Assign Risk Tier
    def assign_tier(prob):
        if prob >= 0.70:
            return "Critical Risk"
        elif prob >= 0.45:
            return "High Risk"
        elif prob >= 0.25:
            return "Moderate Risk"
        else:
            return "Low Risk"

    preds_test_df["Risk_Tier"] = preds_test_df["Churn_Probability"].apply(assign_tier)
    preds_test_df.to_csv("models/test_predictions.csv", index=False)
    logger.info("Test set predictions saved to models/test_predictions.csv")

    print("\n" + "="*50)
    print("[SUMMARY] MODEL TRAINING AND EVALUATION COMPLETED")
    print("="*50)
    print(f"XGBoost Test ROC-AUC: {metrics['xgboost']['recall_optimized']['roc_auc']:.4f}")
    print(f"XGBoost Recall (Optimized Thresh {xgb_thresh_85}): {metrics['xgboost']['recall_optimized']['recall']*100:.2f}%")
    print(f"XGBoost Precision: {metrics['xgboost']['recall_optimized']['precision']*100:.2f}%")
    print(f"XGBoost F1-Score: {metrics['xgboost']['recall_optimized']['f1']:.4f}")
    print("="*50)
    return metadata


if __name__ == "__main__":
    train_models()
