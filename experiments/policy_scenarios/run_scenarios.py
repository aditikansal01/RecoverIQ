"""
Run each constructed policy scenario through the REAL orchestrator (agent + policy +
simulator) and verify the expected rule actually fires. These are targeted correctness
checks, not the fair evaluation sample -- kept completely separate.

Usage:
    python run_scenarios.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend", "orchestrator"))
from orchestrator import process_payment  # noqa: E402

OUT_DIR = os.path.dirname(__file__)

SCENARIOS = [
    {
        "payment_id": "SCEN_HIGHVALUE",
        "expected_rule": "high_value_review",
        "expected_final_status": "MANUAL_REVIEW",
        "description": "Rs 75,000 payment must be routed to human review, regardless of recommended action.",
    },
    {
        "payment_id": "SCEN_NOCONSENT",
        "expected_rule": "consent_required",
        "expected_final_status": None,  # don't assert final status -- just that a block happened
        "description": "Customer opted out of communication; agent (blind to consent) should get "
                        "PAYMENT_LINK or REMINDER blocked, then replan to a non-communication action.",
    },
    {
        "payment_id": "SCEN_MAXRETRY",
        "expected_rule": "max_retries",
        "expected_final_status": None,
        "description": "retry_count already at the limit (3); any RETRY_NOW/RETRY_LATER "
                        "recommendation must be blocked, forcing a non-retry action.",
    },
]


def check_scenario(scenario, result) -> dict:
    audit = result.get("audit_trail", [])
    fired_rules = [e["payload"].get("rule") for e in audit if e.get("event_type") == "POLICY_BLOCKED"]
    fired_rules += [e["payload"].get("rule") for e in audit if e.get("event_type") == "POLICY_REVIEW"]

    rule_fired = scenario["expected_rule"] in fired_rules
    status_ok = (scenario["expected_final_status"] is None or
                 result.get("final_status") == scenario["expected_final_status"])

    return {
        "payment_id": scenario["payment_id"],
        "description": scenario["description"],
        "expected_rule": scenario["expected_rule"],
        "rule_fired": rule_fired,
        "final_status": result.get("final_status"),
        "expected_final_status": scenario["expected_final_status"],
        "status_matches_expectation": status_ok,
        "PASS": rule_fired and status_ok,
    }


if __name__ == "__main__":
    all_results = []
    checks = []

    for scenario in SCENARIOS:
        print(f"\n{'='*70}\nSCENARIO: {scenario['payment_id']}\n{scenario['description']}\n{'='*70}")
        result = process_payment(scenario["payment_id"], verbose=True)
        all_results.append(result)
        check = check_scenario(scenario, result)
        checks.append(check)
        print(f"\n>>> {'PASS' if check['PASS'] else 'FAIL'}: expected rule "
              f"'{scenario['expected_rule']}' fired = {check['rule_fired']}")

    with open(os.path.join(OUT_DIR, "scenario_results.json"), "w") as f:
        json.dump({"results": all_results, "checks": checks}, f, indent=2, default=str)

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    for c in checks:
        status = "PASS" if c["PASS"] else "FAIL"
        print(f"[{status}] {c['payment_id']}: rule '{c['expected_rule']}' fired={c['rule_fired']}, "
              f"final_status={c['final_status']}")
    print(f"\nSaved full results to scenario_results.json")