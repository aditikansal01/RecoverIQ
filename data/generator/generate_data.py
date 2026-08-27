"""
RecoverIQ synthetic data generator.

Generates a realistic-feeling payment-failure dataset with actual structure:
failure type, customer history, and retry count all affect recovery probability
for each candidate action. This gives the ML model real signal to learn from,
instead of random noise.

Usage:
    python generate_data.py

Outputs (in data/generated/):
    customers.csv
    merchants.csv
    payments.csv          -- historical, used for training
    payments_eval.csv     -- held-out evaluation batch, used for final reporting
"""

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 2000
N_MERCHANTS = 40
N_HISTORICAL_PAYMENTS = 5000
N_EVAL_PAYMENTS = 1000

FAILURE_REASONS = [
    "BANK_TIMEOUT",
    "EXPIRED_CARD",
    "INSUFFICIENT_FUNDS",
    "NETWORK_ERROR",
    "MANDATE_AUTH_FAILED",
    "CHECKOUT_ABANDONED",
    "UNKNOWN",
]
# Roughly how common each failure type is in the wild
FAILURE_WEIGHTS = [0.22, 0.16, 0.20, 0.14, 0.10, 0.13, 0.05]

PAYMENT_METHODS = ["UPI_AUTOPAY", "CARD", "UPI", "NETBANKING"]

ACTIONS = ["RETRY_NOW", "RETRY_LATER", "PAYMENT_LINK", "REMINDER", "ESCALATE"]

# Base recovery probability of each action, given failure reason.
# These encode real domain intuition (see conversation for RBI/UPI AutoPay sourcing):
#   - temporary/technical failures (bank timeout, network error) respond well to retry
#   - expired card / mandate auth failures need a fresh instrument, not a blind retry
#   - insufficient funds responds poorly to an immediate retry, better after a delay
#   - checkout abandonment is not really a "failure" -> reminder/link, not retry
BASE_PROB = {
    "BANK_TIMEOUT":        {"RETRY_NOW": 0.35, "RETRY_LATER": 0.72, "PAYMENT_LINK": 0.55, "REMINDER": 0.30, "ESCALATE": 0.95},
    "EXPIRED_CARD":        {"RETRY_NOW": 0.04, "RETRY_LATER": 0.06, "PAYMENT_LINK": 0.68, "REMINDER": 0.40, "ESCALATE": 0.90},
    "INSUFFICIENT_FUNDS":  {"RETRY_NOW": 0.12, "RETRY_LATER": 0.48, "PAYMENT_LINK": 0.38, "REMINDER": 0.35, "ESCALATE": 0.85},
    "NETWORK_ERROR":       {"RETRY_NOW": 0.61, "RETRY_LATER": 0.58, "PAYMENT_LINK": 0.45, "REMINDER": 0.25, "ESCALATE": 0.92},
    "MANDATE_AUTH_FAILED": {"RETRY_NOW": 0.05, "RETRY_LATER": 0.10, "PAYMENT_LINK": 0.60, "REMINDER": 0.30, "ESCALATE": 0.88},
    "CHECKOUT_ABANDONED":  {"RETRY_NOW": 0.02, "RETRY_LATER": 0.05, "PAYMENT_LINK": 0.42, "REMINDER": 0.51, "ESCALATE": 0.60},
    "UNKNOWN":             {"RETRY_NOW": 0.20, "RETRY_LATER": 0.28, "PAYMENT_LINK": 0.30, "REMINDER": 0.22, "ESCALATE": 0.75},
}

# Deterministic baseline strategy (rule-based, no AI) -- this is our control group.
def baseline_action(failure_reason: str, retry_count: int) -> str:
    if retry_count >= 3:
        return "ESCALATE"
    if failure_reason == "BANK_TIMEOUT":
        return "RETRY_LATER"
    if failure_reason == "EXPIRED_CARD":
        return "PAYMENT_LINK"
    if failure_reason == "INSUFFICIENT_FUNDS":
        return "RETRY_LATER"
    if failure_reason == "NETWORK_ERROR":
        return "RETRY_NOW"
    if failure_reason == "MANDATE_AUTH_FAILED":
        return "PAYMENT_LINK"
    if failure_reason == "CHECKOUT_ABANDONED":
        return "REMINDER"
    return "REMINDER"


def make_merchants(n):
    rows = []
    categories = ["ecommerce", "subscription", "edtech", "travel", "food_delivery", "saas"]
    for i in range(n):
        rows.append({
            "merchant_id": f"M{i:04d}",
            "merchant_name": fake.company(),
            "category": random.choice(categories),
            "avg_transaction": round(np.random.gamma(2.0, 800), 2),
        })
    return pd.DataFrame(rows)


def make_customers(n):
    rows = []
    for i in range(n):
        account_age = int(np.random.exponential(220))
        # more tenure -> generally more successful payment history
        successes = max(0, int(np.random.poisson(1 + account_age / 90)))
        failures = max(0, int(np.random.poisson(1.2)))
        rows.append({
            "customer_id": f"C{i:05d}",
            "account_age_days": account_age,
            "previous_successes": successes,
            "previous_failures": failures,
            "avg_payment_amount": round(np.random.gamma(2.2, 700), 2),
            "consent_status": random.random() > 0.06,  # ~6% opted out of comms
            "preferred_payment_method": random.choice(PAYMENT_METHODS),
        })
    return pd.DataFrame(rows)


def recovery_prob(failure_reason, action, retry_count, previous_successes, previous_failures, account_age_days):
    """The 'true' underlying probability used to simulate outcomes. The ML model
    never sees this function -- it only sees features and observed outcomes, and has
    to learn an approximation of it from data, same as in a real system."""
    p = BASE_PROB[failure_reason][action]

    # more retries on the same payment -> diminishing returns (fatigue / already-tried signals)
    p *= max(0.25, 1 - 0.22 * retry_count)

    # stronger customer history -> modest positive adjustment
    history_score = previous_successes - previous_failures
    p *= np.clip(1 + 0.015 * history_score, 0.6, 1.25)

    # very new accounts recover slightly worse (less trust/verified instruments)
    if account_age_days < 14:
        p *= 0.85

    return float(np.clip(p, 0.01, 0.98))


def make_payments(customers_df, merchants_df, n, is_eval=False):
    rows = []
    start_date = datetime(2026, 6, 1)
    for i in range(n):
        cust = customers_df.sample(1).iloc[0]
        merch = merchants_df.sample(1).iloc[0]
        failure_reason = random.choices(FAILURE_REASONS, weights=FAILURE_WEIGHTS, k=1)[0]

        # amount tends to track the customer's typical spend, with noise
        amount = round(max(99, np.random.normal(cust["avg_payment_amount"], cust["avg_payment_amount"] * 0.35)), 2)

        retry_count = np.random.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.13, 0.07])
        created_at = start_date + timedelta(days=random.randint(0, 85), hours=random.randint(0, 23))

        action = baseline_action(failure_reason, retry_count)
        prob = recovery_prob(
            failure_reason, action, retry_count,
            cust["previous_successes"], cust["previous_failures"], cust["account_age_days"]
        )
        recovered = np.random.random() < prob
        status = "RECOVERED" if recovered else ("UNRECOVERABLE" if retry_count >= 3 else "FAILED")

        prefix = "E" if is_eval else "P"
        rows.append({
            "payment_id": f"{prefix}{i:05d}",
            "merchant_id": merch["merchant_id"],
            "customer_id": cust["customer_id"],
            "amount": amount,
            "payment_method": cust["preferred_payment_method"],
            "status": status,
            "failure_reason": failure_reason,
            "retry_count": int(retry_count),
            "created_at": created_at.isoformat(),
            "last_attempt_at": (created_at + timedelta(hours=int(retry_count) * 6)).isoformat(),
            "is_evaluation_set": is_eval,
            # ground-truth labels kept ONLY for training/eval scripts, not for the agent at inference time
            "baseline_action": action,
            "baseline_recovery_prob_true": round(prob, 4),
            "baseline_recovered": bool(recovered),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "generated")
    os.makedirs(out_dir, exist_ok=True)

    merchants = make_merchants(N_MERCHANTS)
    customers = make_customers(N_CUSTOMERS)
    payments = make_payments(customers, merchants, N_HISTORICAL_PAYMENTS, is_eval=False)
    payments_eval = make_payments(customers, merchants, N_EVAL_PAYMENTS, is_eval=True)

    merchants.to_csv(os.path.join(out_dir, "merchants.csv"), index=False)
    customers.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    payments.to_csv(os.path.join(out_dir, "payments.csv"), index=False)
    payments_eval.to_csv(os.path.join(out_dir, "payments_eval.csv"), index=False)

    print(f"Generated {len(merchants)} merchants, {len(customers)} customers")
    print(f"Generated {len(payments)} historical payments, {len(payments_eval)} evaluation payments")
    print("\nFailure reason distribution (historical):")
    print(payments["failure_reason"].value_counts())
    print("\nBaseline recovery rate by failure reason (historical):")
    print(payments.groupby("failure_reason")["baseline_recovered"].mean().round(3))
    print(f"\nOverall baseline recovery rate: {payments['baseline_recovered'].mean():.3f}")
    print(f"Overall baseline recovered amount: Rs {payments.loc[payments['baseline_recovered'], 'amount'].sum():,.2f}")
    print(f"Overall revenue at risk (historical set): Rs {payments['amount'].sum():,.2f}")