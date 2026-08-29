"""
Step 2 of Day 6: run the full agent + policy + replanning pipeline across the
sample selected by select_sample.py.

RESUMABLE: results are appended to ai_batch_results.jsonl one payment at a time.
If you hit a quota limit (429) partway through, just re-run this script later --
it skips payments already in the output file and continues from where it stopped.

Usage:
    python run_ai_batch.py
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend", "orchestrator"))
from orchestrator import process_payment  # noqa: E402

OUT_DIR = os.path.dirname(__file__)
SAMPLE_PATH = os.path.join(OUT_DIR, "eval_sample.csv")
RESULTS_PATH = os.path.join(OUT_DIR, "ai_batch_results.jsonl")

DELAY_BETWEEN_PAYMENTS_SECONDS = 8  # be gentle on RPM limits


def already_processed_ids() -> set:
    if not os.path.exists(RESULTS_PATH):
        return set()
    ids = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["payment_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


if __name__ == "__main__":
    if not os.path.exists(SAMPLE_PATH):
        print("Run select_sample.py first.")
        sys.exit(1)

    sample = pd.read_csv(SAMPLE_PATH)
    done = already_processed_ids()
    remaining = sample[~sample["payment_id"].isin(done)]

    print(f"Sample size: {len(sample)}   Already done: {len(done)}   Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All payments already processed. Run summarize_results.py.")
        sys.exit(0)

    processed_this_run = 0
    with open(RESULTS_PATH, "a") as out_f:
        for i, row in remaining.iterrows():
            payment_id = row["payment_id"]
            print(f"\n{'='*60}\nProcessing {payment_id} ({processed_this_run + 1}/{len(remaining)} this run, "
                  f"{len(done) + processed_this_run + 1}/{len(sample)} total)\n{'='*60}")
            try:
                # deterministic per-payment seed so re-running a payment (if ever needed)
                # gives the same simulated outcome
                seed = abs(hash(payment_id)) % (2**31)
                result = process_payment(payment_id, verbose=True, sim_seed=seed)
                out_f.write(json.dumps(result, default=str) + "\n")
                out_f.flush()
                processed_this_run += 1
            except Exception as e:
                error_msg = str(e)
                print(f"\n!!! STOPPED at {payment_id}: {error_msg}")
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    print(f"\nQuota limit hit. Processed {processed_this_run} payments this run "
                          f"({len(done) + processed_this_run}/{len(sample)} total).")
                    print("Progress is saved. Re-run this script later (e.g. tomorrow, or after "
                          "the quota resets) to continue from here.")
                else:
                    print("Unexpected error -- progress up to this point is saved. "
                          "Investigate before re-running.")
                sys.exit(1)

            time.sleep(DELAY_BETWEEN_PAYMENTS_SECONDS)

    print(f"\nAll {len(sample)} payments processed. Run summarize_results.py for the final comparison.")