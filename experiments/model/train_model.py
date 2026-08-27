"""
Train logistic regression and XGBoost on the historical dataset, evaluate both on a
held-out test split, and pick whichever genuinely performs better. If XGBoost doesn't
show a meaningful improvement, we ship logistic regression -- simpler and more
explainable, and the honest choice given a dataset this size.

Usage:
    python train_model.py
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
from ml.features import build_dataset, FEATURE_COLUMNS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "ml")
OUT_DIR = os.path.dirname(__file__)

MEANINGFUL_LIFT_THRESHOLD = 0.03  # XGBoost must beat logistic regression's ROC-AUC by
                                    # at least this much to be worth the extra complexity


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }


if __name__ == "__main__":
    payments = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))

    X, y = build_dataset(payments, customers)

    # 70% train, 15% validation, 15% held-out test -- test set is touched exactly once, at the end.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_metrics = evaluate(logreg, X_test_scaled, y_test)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate(xgb, X_test, y_test)

    print("\nLogistic Regression:", logreg_metrics)
    print("XGBoost:            ", xgb_metrics)

    lift = xgb_metrics["roc_auc"] - logreg_metrics["roc_auc"]
    chosen = "xgboost" if lift >= MEANINGFUL_LIFT_THRESHOLD else "logistic_regression"

    print(f"\nROC-AUC lift from XGBoost: {lift:+.4f} (threshold for adoption: {MEANINGFUL_LIFT_THRESHOLD})")
    print(f"Chosen model: {chosen}")

    if chosen == "xgboost":
        joblib.dump(xgb, os.path.join(MODEL_DIR, "recovery_model.joblib"))
        joblib.dump(None, os.path.join(MODEL_DIR, "recovery_scaler.joblib"))
    else:
        joblib.dump(logreg, os.path.join(MODEL_DIR, "recovery_model.joblib"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "recovery_scaler.joblib"))

    with open(os.path.join(MODEL_DIR, "model_type.json"), "w") as f:
        json.dump({"chosen_model": chosen}, f)

    comparison = {
        "logistic_regression": logreg_metrics,
        "xgboost": xgb_metrics,
        "roc_auc_lift": round(float(lift), 4),
        "threshold_for_adoption": MEANINGFUL_LIFT_THRESHOLD,
        "chosen_model": chosen,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
    with open(os.path.join(OUT_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved model to backend/ml/recovery_model.joblib")
    print(f"Saved comparison to experiments/model/model_comparison.json")