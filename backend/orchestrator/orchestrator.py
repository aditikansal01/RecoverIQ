"""
Orchestrator -- runs the full RecoverIQ loop for a single payment:

    agent recommends -> policy engine gates -> if BLOCKED, agent replans (up to a
    limit) -> if ALLOWED, simulator executes -> everything is logged to an audit trail

This is the piece that turns "an LLM with tools" into "AI recommends, deterministic
policy controls, agent adapts to rejection" -- the core differentiator of the project.

Usage:
    python orchestrator.py PAYMENT_ID
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "policy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import agent as agent_module  # noqa: E402
import tools  # noqa: E402
from policy_engine import evaluate_policy  # noqa: E402
from payment_simulator import PaymentSimulator  # noqa: E402

MAX_REPLAN_ATTEMPTS = 3


def _now():
    return datetime.now(timezone.utc).isoformat()


def process_payment(payment_id: str, verbose: bool = True, sim_seed: int | None = None) -> dict:
    audit_trail = []

    def log(event_type, payload):
        entry = {"timestamp": _now(), "event_type": event_type, "payload": payload}
        audit_trail.append(entry)
        if verbose:
            print(f"[{event_type}] {payload}")

    payment = tools.get_payment_details(payment_id)
    if "error" in payment:
        return {"payment_id": payment_id, "final_status": "ERROR", "error": payment["error"]}
    customer = tools.get_customer_history(payment["customer_id"])
    if "error" in customer:
        return {"payment_id": payment_id, "final_status": "ERROR", "error": customer["error"]}

    log("PAYMENT_FAILED", {"payment_id": payment_id, "amount": payment["amount"],
                            "failure_reason": payment["failure_reason"]})

    excluded_actions = []
    rejection_reason = None
    sim = PaymentSimulator(technical_failure_rate=0.04, seed=sim_seed)

    for attempt in range(1, MAX_REPLAN_ATTEMPTS + 1):
        log("AI_ANALYSIS_STARTED", {"attempt": attempt, "excluded_actions": list(excluded_actions)})

        recommendation = agent_module.run_agent(
            payment_id, verbose=verbose,
            excluded_actions=excluded_actions or None,
            rejection_reason=rejection_reason,
        )
        if "error" in recommendation:
            log("AGENT_ERROR", recommendation)
            return {"payment_id": payment_id, "final_status": "AGENT_ERROR",
                    "audit_trail": audit_trail, "error": recommendation}

        action = recommendation["recommended_action"]
        log("AI_RECOMMENDATION", {"action": action, "expected_value": recommendation.get("expected_value"),
                                   "confidence": recommendation.get("confidence"),
                                   "reasoning": recommendation.get("reasoning")})

        decision = evaluate_policy(payment, customer, action, previously_tried_actions=excluded_actions)

        if decision["decision"] == "BLOCK":
            log("POLICY_BLOCKED", {"action": action, "rule": decision["rule"], "reason": decision["reason"]})
            excluded_actions.append(action)
            rejection_reason = decision["reason"]
            log("AGENT_REPLANNED", {"attempt": attempt, "excluded_so_far": list(excluded_actions)})
            continue

        if decision["decision"] == "REVIEW":
            log("POLICY_REVIEW", {"action": action, "rule": decision["rule"], "reason": decision["reason"]})
            return {
                "payment_id": payment_id, "final_status": "MANUAL_REVIEW",
                "recommended_action": action, "policy_reason": decision["reason"],
                "audit_trail": audit_trail,
            }

        # ALLOW -> execute
        log("POLICY_APPROVED", {"action": action, "rule": decision["rule"]})
        sim_payment = {
            "failure_reason": payment["failure_reason"], "retry_count": payment["retry_count"],
            "previous_successes": customer["previous_successes"], "previous_failures": customer["previous_failures"],
            "account_age_days": customer["account_age_days"], "amount": payment["amount"],
        }
        outcome = sim.execute_action(sim_payment, action)
        log("ACTION_EXECUTED", {"action": action, "execution_status": outcome["execution_status"]})

        if outcome["execution_status"] == "TIMEOUT":
            log("RECOVERY_FAILED", {"reason": "technical timeout, not a customer decline"})
            # a technical failure is different from "customer didn't pay" -- give it
            # one more attempt via replanning rather than silently giving up
            excluded_actions.append(action)
            rejection_reason = "Previous execution attempt hit a technical timeout (not a customer decline)."
            log("AGENT_REPLANNED", {"attempt": attempt, "reason": "technical_timeout"})
            continue

        final_status = "RECOVERED" if outcome["recovered"] else "UNRECOVERED"
        log("PAYMENT_RECOVERED" if outcome["recovered"] else "RECOVERY_FAILED",
            {"recovered_amount": outcome["recovered_amount"]})

        return {
            "payment_id": payment_id, "final_status": final_status, "action_taken": action,
            "recovered": outcome["recovered"], "recovered_amount": outcome["recovered_amount"],
            "attempts_needed": attempt, "audit_trail": audit_trail,
        }

    # exhausted replan attempts without resolution -> escalate
    log("ESCALATE", {"reason": f"exhausted {MAX_REPLAN_ATTEMPTS} replan attempts without an allowed action"})
    return {
        "payment_id": payment_id, "final_status": "ESCALATED",
        "audit_trail": audit_trail,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python orchestrator.py PAYMENT_ID")
        sys.exit(1)

    result = process_payment(sys.argv[1])
    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2, default=str))