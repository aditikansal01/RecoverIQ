"""
RecoverIQ dashboard API. Serves the real evaluation results (baseline vs AI system)
and the static dashboard frontend.

Usage:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(__file__)
EVAL_DIR = os.path.join(BASE_DIR, "..", "..", "experiments", "evaluation")
BASELINE_FULL_PATH = os.path.join(BASE_DIR, "..", "..", "experiments", "baseline", "baseline_results.json")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

app = FastAPI(title="RecoverIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load_ai_results() -> list[dict]:
    path = os.path.join(EVAL_DIR, "ai_batch_results.jsonl")
    if not os.path.exists(path):
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def _load_sample_amounts() -> dict:
    path = os.path.join(EVAL_DIR, "eval_sample.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["payment_id"], zip(df["amount"], df["failure_reason"])))


@app.get("/api/summary")
def get_summary():
    ai_results = _load_ai_results()
    sample_info = _load_sample_amounts()
    sample_size = len(sample_info) if sample_info else None

    baseline_sample = {}
    baseline_sample_path = os.path.join(EVAL_DIR, "baseline_sample_results.json")
    if os.path.exists(baseline_sample_path):
        with open(baseline_sample_path) as f:
            baseline_sample = json.load(f)

    baseline_full = {}
    if os.path.exists(BASELINE_FULL_PATH):
        with open(BASELINE_FULL_PATH) as f:
            baseline_full = json.load(f)

    processed = len(ai_results)
    recovered_amount = sum(r.get("recovered_amount", 0) or 0 for r in ai_results)
    recovered_count = sum(1 for r in ai_results if r.get("recovered"))
    manual_review = sum(1 for r in ai_results if r.get("final_status") == "MANUAL_REVIEW")
    escalated = sum(1 for r in ai_results if r.get("final_status") == "ESCALATED")
    errors = sum(1 for r in ai_results if r.get("final_status") in ("ERROR", "AGENT_ERROR"))
    replanned = sum(1 for r in ai_results if r.get("attempts_needed", 1) > 1)
    policy_blocks = sum(
        1 for r in ai_results for e in r.get("audit_trail", [])
        if isinstance(e, dict) and e.get("event_type") == "POLICY_BLOCKED"
    )
    revenue_at_risk_processed = sum(
        sample_info.get(r["payment_id"], (0, None))[0] for r in ai_results
    )

    baseline_recovered_prorated = None
    if baseline_sample and processed:
        baseline_recovered_prorated = round(
            baseline_sample.get("recovered_amount", 0) * (processed / sample_size) if sample_size else 0, 2
        )

    return {
        "sample_size": sample_size,
        "processed": processed,
        "revenue_at_risk": round(revenue_at_risk_processed, 2),
        "ai_recovered_amount": round(recovered_amount, 2),
        "ai_recovery_rate": round(recovered_count / processed, 4) if processed else None,
        "baseline_recovered_amount_full_1000": baseline_full.get("recovered_amount"),
        "baseline_recovery_rate_full_1000": baseline_full.get("recovery_rate"),
        "baseline_recovered_amount_same_sample_prorated": baseline_recovered_prorated,
        "manual_review_count": manual_review,
        "escalated_count": escalated,
        "error_count": errors,
        "replanned_count": replanned,
        "policy_block_count": policy_blocks,
    }


@app.get("/api/payments")
def get_payments():
    ai_results = _load_ai_results()
    sample_info = _load_sample_amounts()
    payments = []
    for r in ai_results:
        pid = r.get("payment_id")
        amount, failure_reason = sample_info.get(pid, (r.get("amount", 0), None))
        payments.append({
            "payment_id": pid,
            "amount": amount,
            "failure_reason": failure_reason,
            "final_status": r.get("final_status"),
            "action_taken": r.get("action_taken"),
            "recovered": r.get("recovered"),
            "recovered_amount": r.get("recovered_amount"),
            "attempts_needed": r.get("attempts_needed", 1),
        })
    return payments


@app.get("/api/payment/{payment_id}")
def get_payment_detail(payment_id: str):
    ai_results = _load_ai_results()
    for r in ai_results:
        if r.get("payment_id") == payment_id:
            return r
    raise HTTPException(status_code=404, detail="Payment not found in processed results")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")