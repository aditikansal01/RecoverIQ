"""
Feature engineering for the recovery-probability model.

predict_recovery_probability() is the exact function the agent will call as a tool
in Day 4 -- it takes a payment + a candidate action and returns a probability the
agent can reason over, instead of the LLM guessing a number.
"""

import numpy as np
import pandas as pd

FAILURE_REASONS = [
    "BANK_TIMEOUT", "EXPIRED_CARD", "INSUFFICIENT_FUNDS",
    "NETWORK_ERROR", "MANDATE_AUTH_FAILED", "CHECKOUT_ABANDONED", "UNKNOWN",
]
ACTIONS = ["RETRY_NOW", "RETRY_LATER", "PAYMENT_LINK", "REMINDER", "ESCALATE"]

FEATURE_COLUMNS = (
    [f"failure_{r}" for r in FAILURE_REASONS]
    + [f"action_{a}" for a in ACTIONS]
    + ["retry_count", "previous_successes", "previous_failures", "account_age_days",
       "amount", "history_score"]
)


def build_feature_row(failure_reason, action, retry_count, previous_successes,
                       previous_failures, account_age_days, amount) -> dict:
    row = {f"failure_{r}": int(r == failure_reason) for r in FAILURE_REASONS}
    row.update({f"action_{a}": int(a == action) for a in ACTIONS})
    row["retry_count"] = retry_count
    row["previous_successes"] = previous_successes
    row["previous_failures"] = previous_failures
    row["account_age_days"] = account_age_days
    row["amount"] = amount
    row["history_score"] = previous_successes - previous_failures
    return row


def build_dataset(payments_df: pd.DataFrame, customers_df: pd.DataFrame):
    """Returns (X, y) for training. Uses the action actually taken (baseline_action,
    including exploration) and the observed outcome (baseline_recovered) as labels."""
    customers_indexed = customers_df.set_index("customer_id")
    rows = []
    labels = []
    for _, r in payments_df.iterrows():
        cust = customers_indexed.loc[r["customer_id"]]
        rows.append(build_feature_row(
            failure_reason=r["failure_reason"],
            action=r["baseline_action"],
            retry_count=r["retry_count"],
            previous_successes=cust["previous_successes"],
            previous_failures=cust["previous_failures"],
            account_age_days=cust["account_age_days"],
            amount=r["amount"],
        ))
        labels.append(int(r["baseline_recovered"]))
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = pd.Series(labels, name="recovered")
    return X, y