"""
predict_recovery_probability -- the tool the agent (Day 4) calls to get a real,
data-backed probability estimate instead of the LLM inventing a number.

Loads whichever model train_model.py chose (logistic regression or XGBoost) and
exposes one clean function.
"""

import json
import os

import joblib
import pandas as pd

from features import build_feature_row, FEATURE_COLUMNS

_DIR = os.path.dirname(__file__)

with open(os.path.join(_DIR, "model_type.json")) as f:
    MODEL_TYPE = json.load(f)["chosen_model"]

_model = joblib.load(os.path.join(_DIR, "recovery_model.joblib"))
_scaler = joblib.load(os.path.join(_DIR, "recovery_scaler.joblib"))  # None if XGBoost


def predict_recovery_probability(failure_reason: str, action: str, retry_count: int,
                                  previous_successes: int, previous_failures: int,
                                  account_age_days: int, amount: float) -> float:
    row = build_feature_row(failure_reason, action, retry_count, previous_successes,
                             previous_failures, account_age_days, amount)
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    if _scaler is not None:
        X = _scaler.transform(X)
    return float(_model.predict_proba(X)[0, 1])


if __name__ == "__main__":
    # quick sanity check
    p = predict_recovery_probability(
        failure_reason="BANK_TIMEOUT", action="RETRY_LATER", retry_count=0,
        previous_successes=5, previous_failures=1, account_age_days=300, amount=4999,
    )
    print(f"Model in use: {MODEL_TYPE}")
    print(f"Example prediction -- P(recovery | BANK_TIMEOUT, RETRY_LATER): {p:.3f}")