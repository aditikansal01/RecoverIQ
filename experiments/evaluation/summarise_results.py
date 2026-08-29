"""
Step 3 of Day 6: summarize AI batch results and compare against baseline (both
computed on the SAME sample, for a fair comparison). Safe to run on partial
results -- just reports how many payments have been processed so far.

Usage:
    python summarize_results.py
"""

import json
import os

OUT_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(OUT_DIR, "ai_batch_results.jsonl")
BASELINE_PATH = os.path.join(OUT_DIR, "baseline_sample_results.json")
SAMPLE_PATH = os.path.join(OUT_DIR, "eval_sample.csv")


if __name__ == "__main__":
    import pandas as pd

    if not os.path.exists(RESULTS_PATH):
        print("No results yet -- run run_ai_batch.py first.")
        exit(1)

    sample = pd.read_csv(SAMPLE_PATH)
    amount_by_id = dict(zip(sample["payment_id"], sample["amount"]))

    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    total_in_sample = len(sample)
    processed = len(results)

    recovered_amount = sum(r.get("recovered_amount", 0) or 0 for r in results)
    recovered_count = sum(1 for r in results if r.get("recovered"))
    manual_review_count = sum(1 for r in results if r.get("final_status") == "MANUAL_REVIEW")
    escalated_count = sum(1 for r in results if r.get("final_status") == "ESCALATED")
    error_count = sum(1 for r in results if r.get("final_status") in ("ERROR", "AGENT_ERROR"))

    replanned_count = sum(1 for r in results if r.get("attempts_needed", 1) > 1)
    policy_blocks = sum(
        1 for r in results for e in r.get("audit_trail", [])
        if isinstance(e, dict) and e.get("event_type") == "POLICY_BLOCKED"
    )

    revenue_at_risk_processed = sum(amount_by_id.get(r["payment_id"], 0) for r in results)

    ai_summary = {
        "sample_size": total_in_sample,
        "processed_so_far": processed,
        "revenue_at_risk_in_processed_subset": round(revenue_at_risk_processed, 2),
        "recovered_amount": round(recovered_amount, 2),
        "recovery_rate": round(recovered_count / processed, 4) if processed else None,
        "manual_review_count": manual_review_count,
        "escalated_count": escalated_count,
        "error_count": error_count,
        "payments_that_needed_replanning": replanned_count,
        "total_policy_blocks": policy_blocks,
    }

    print("=== AI SYSTEM (agent + policy + replanning) ===")
    print(json.dumps(ai_summary, indent=2))

    if processed < total_in_sample:
        print(f"\n({total_in_sample - processed} payments not yet processed -- "
              f"re-run run_ai_batch.py to continue. Numbers below reflect only what's done so far.)")

    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print("\n=== BASELINE (rule-based, same full sample) ===")
        print(json.dumps(baseline, indent=2))

        if processed > 0:
            improvement = recovered_amount - baseline["recovered_amount"] * (processed / total_in_sample)
            print(f"\n=== COMPARISON (on {processed} processed payments so far) ===")
            print(f"AI recovered:       Rs {recovered_amount:,.2f}")
            print(f"Baseline recovered (prorated to same subset): ~Rs "
                  f"{baseline['recovered_amount'] * (processed / total_in_sample):,.2f}")
            print(f"Difference:          Rs {improvement:,.2f}")
            print("\nNote: for the final, official number, wait until processed == sample_size, "
                  "then this comparison uses the exact same payments for both systems.")

    with open(os.path.join(OUT_DIR, "ai_summary.json"), "w") as f:
        json.dump(ai_summary, f, indent=2)