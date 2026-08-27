"""
Evaluate the deterministic BASELINE strategy over the held-out evaluation set.

This produces the control-group numbers. Every later system (ML-driven, then
agent-driven) must be compared against these same numbers on the same batch --
never against a cherry-picked subset.

Usage:
    python evaluate_baseline.py
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
from simulator.payment_simulator import PaymentSimulator, baseline_action  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
OUT_DIR = os.path.dirname(__file__)


def run_baseline(eval_df: pd.DataFrame, sim: PaymentSimulator) -> pd.DataFrame:
    records = []
    for _, row in eval_df.iterrows():
        payment = {
            "failure_reason": row["failure_reason"],
            "retry_count": int(row["retry_count"]),
            "previous_successes": 0,   # filled in below from customers.csv
            "previous_failures": 0,
            "account_age_days": 0,
            "amount": row["amount"],
        }
        records.append(payment)
    return records


if __name__ == "__main__":
    payments_eval = pd.read_csv(os.path.join(DATA_DIR, "payments_eval.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv")).set_index("customer_id")

    sim = PaymentSimulator(technical_failure_rate=0.04, seed=7)  # different seed from generation -> honest re-simulation

    results = []
    for _, row in payments_eval.iterrows():
        cust = customers.loc[row["customer_id"]]
        payment = {
            "failure_reason": row["failure_reason"],
            "retry_count": int(row["retry_count"]),
            "previous_successes": int(cust["previous_successes"]),
            "previous_failures": int(cust["previous_failures"]),
            "account_age_days": int(cust["account_age_days"]),
            "amount": float(row["amount"]),
        }
        action = baseline_action(payment["failure_reason"], payment["retry_count"])
        outcome = sim.execute_action(payment, action)

        results.append({
            "payment_id": row["payment_id"],
            "amount": payment["amount"],
            "failure_reason": payment["failure_reason"],
            "action_taken": action,
            "execution_status": outcome["execution_status"],
            "recovered": outcome["recovered"],
            "recovered_amount": outcome["recovered_amount"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "baseline_run_detail.csv"), index=False)

    revenue_at_risk = results_df["amount"].sum()
    recovered_amount = results_df["recovered_amount"].sum()
    recovery_rate = results_df["recovered"].mean()
    timeout_rate = (results_df["execution_status"] == "TIMEOUT").mean()

    summary = {
        "payments_evaluated": int(len(results_df)),
        "revenue_at_risk": round(float(revenue_at_risk), 2),
        "recovered_amount": round(float(recovered_amount), 2),
        "recovery_rate": round(float(recovery_rate), 4),
        "technical_timeout_rate": round(float(timeout_rate), 4),
        "action_distribution": results_df["action_taken"].value_counts().to_dict(),
        "recovery_rate_by_failure_reason": results_df.groupby("failure_reason")["recovered"].mean().round(4).to_dict(),
    }

    with open(os.path.join(OUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved detail to baseline_run_detail.csv and summary to baseline_results.json")
    print("These are the numbers the AI system (Day 3 onward) must beat.")