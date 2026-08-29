"""
Step 1 of Day 6: pick a representative sample of the evaluation set and compute
the baseline strategy's result on that EXACT sample -- so later, when we compare
the AI system against the baseline, both ran on the identical set of payments.

No API calls in this script -- safe to run as many times as needed.

Usage:
    python select_sample.py [SAMPLE_SIZE]   (default 100)
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend", "simulator"))
from payment_simulator import PaymentSimulator, baseline_action  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
OUT_DIR = os.path.dirname(__file__)

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def stratified_sample(payments_eval: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample proportionally from each failure_reason group, so rare failure types
    aren't drowned out or missing entirely -- keeps the sample representative."""
    frac = n / len(payments_eval)
    parts = []
    for _, group in payments_eval.groupby("failure_reason"):
        parts.append(group.sample(frac=frac, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    # trim/pad to get close to exactly n (grouping fractions rarely land exactly on n)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.reset_index(drop=True)


if __name__ == "__main__":
    payments_eval = pd.read_csv(os.path.join(DATA_DIR, "payments_eval.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv")).set_index("customer_id")

    sample = stratified_sample(payments_eval, SAMPLE_SIZE)
    sample.to_csv(os.path.join(OUT_DIR, "eval_sample.csv"), index=False)
    print(f"Selected {len(sample)} payments (stratified by failure_reason).")
    print(sample["failure_reason"].value_counts())

    # Baseline, computed on this exact sample
    sim = PaymentSimulator(technical_failure_rate=0.04, seed=99)
    results = []
    for _, row in sample.iterrows():
        cust = customers.loc[row["customer_id"]]
        payment = {
            "failure_reason": row["failure_reason"], "retry_count": int(row["retry_count"]),
            "previous_successes": int(cust["previous_successes"]), "previous_failures": int(cust["previous_failures"]),
            "account_age_days": int(cust["account_age_days"]), "amount": float(row["amount"]),
        }
        action = baseline_action(payment["failure_reason"], payment["retry_count"])
        outcome = sim.execute_action(payment, action)
        results.append({
            "payment_id": row["payment_id"], "amount": payment["amount"],
            "action_taken": action, "recovered": outcome["recovered"],
            "recovered_amount": outcome["recovered_amount"],
        })

    results_df = pd.DataFrame(results)
    summary = {
        "sample_size": len(results_df),
        "revenue_at_risk": round(float(results_df["amount"].sum()), 2),
        "recovered_amount": round(float(results_df["recovered_amount"].sum()), 2),
        "recovery_rate": round(float(results_df["recovered"].mean()), 4),
    }
    with open(os.path.join(OUT_DIR, "baseline_sample_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBaseline on this sample:")
    print(json.dumps(summary, indent=2))
    print("\nSaved eval_sample.csv and baseline_sample_results.json")
    print("Next: run run_ai_batch.py to run the agent+policy system on this same sample.")