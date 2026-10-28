"""
Feature Engineering and Preprocessing Module for Customer Churn Prediction.
Transforms raw customer data into feature vectors with engineered domain features,
scaling, and one-hot encoding.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Core feature definitions
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class TelcoDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer that adds telecom domain-specific features:
    - num_services: count of active subscribed services
    - avg_monthly_charges: historical average spend per month
    - charge_ratio: ratio of TotalCharges to (tenure * MonthlyCharges)
    - has_internet: binary flag
    - is_month_to_month: high-risk contract indicator
    - has_auto_payment: automatic payment indicator (reduces churn)
    - has_family_support: partner or dependents present
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if isinstance(df, np.ndarray):
            raise ValueError("TelcoDomainFeatureExtractor expects a pandas DataFrame input.")

        # TotalCharges safety check
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # 1. Total active services count
        service_counts = np.zeros(len(df))
        for col in SERVICE_COLUMNS:
            if col in df.columns:
                service_counts += (df[col].astype(str) == "Yes").astype(int)
        df["num_services"] = service_counts

        # 2. Average monthly charges over tenure
        tenure = df["tenure"].astype(float).clip(lower=0)
        df["avg_monthly_charges"] = df["TotalCharges"] / (tenure + 1.0)

        # 3. Charge deviation ratio
        expected_total = tenure * df["MonthlyCharges"].astype(float)
        df["charge_ratio"] = df["TotalCharges"] / (expected_total + 1e-4)

        # 4. Has internet service
        if "InternetService" in df.columns:
            df["has_internet"] = (df["InternetService"].astype(str) != "No").astype(int)
        else:
            df["has_internet"] = 0

        # 5. Month-to-month contract flag (major churn driver)
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)
        else:
            df["is_month_to_month"] = 0

        # 6. Auto-payment flag
        if "PaymentMethod" in df.columns:
            df["has_auto_payment"] = (
                df["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False).astype(int)
            )
        else:
            df["has_auto_payment"] = 0

        # 7. Family anchor (partner or dependents)
        has_partner = (df["Partner"].astype(str) == "Yes").astype(int) if "Partner" in df.columns else 0
        has_dep = (df["Dependents"].astype(str) == "Yes").astype(int) if "Dependents" in df.columns else 0
        df["has_family_support"] = ((has_partner + has_dep) > 0).astype(int)

        return df


def build_preprocessor_pipeline():
    """
    Constructs the end-to-end preprocessing pipeline:
    Domain feature extraction -> Standard Scaling for numeric features -> OneHotEncoding for categoricals.
    """
    domain_extractor = TelcoDomainFeatureExtractor()

    # All numeric features after domain engineering
    all_numeric_cols = NUMERIC_FEATURES + [
        "num_services",
        "avg_monthly_charges",
        "charge_ratio",
        "has_internet",
        "is_month_to_month",
        "has_auto_payment",
        "has_family_support",
    ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric_cols),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(steps=[
        ("domain_features", domain_extractor),
        ("column_transforms", column_preprocessor)
    ])

    return full_pipeline


def get_feature_names(fitted_pipeline) -> list:
    """
    Extracts readable feature names from the fitted preprocessor pipeline.
    """
    col_trans = fitted_pipeline.named_steps["column_transforms"]
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "num_services", "avg_monthly_charges", "charge_ratio",
        "has_internet", "is_month_to_month", "has_auto_payment", "has_family_support"
    ]
    
    cat_encoder = col_trans.named_transformers_["cat"].named_steps["onehot"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    
    return numeric_features + cat_features


def save_preprocessor(pipeline, file_path: str = "models/preprocessor.pkl"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(pipeline, file_path)
    logger.info(f"Preprocessor saved to {file_path}")


def load_preprocessor(file_path: str = "models/preprocessor.pkl"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessor not found at {file_path}. Train model first.")
    return joblib.load(file_path)


if __name__ == "__main__":
    from data_loader import load_and_split_data
    X_train, X_test, y_train, y_test, _ = load_and_split_data()
    pipeline = build_preprocessor_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    feature_names = get_feature_names(pipeline)
    print(f"Transformed X_train shape: {X_train_trans.shape}")
    print(f"Total features extracted: {len(feature_names)}")
    print(f"Sample feature names: {feature_names[:10]}")
    save_preprocessor(pipeline)
