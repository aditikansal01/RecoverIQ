"""
Deliberately constructed edge cases to prove each policy rule actually fires, at the
REAL threshold (Rs 50,000), not a weakened one. Written as standalone files that never
touch payments.csv / payments_eval.csv / customers.csv -- so the fair 100-payment
comparison and the full-1000 baseline reference number are never contaminated.

Usage:
    python scenario_data.py
"""

import os

import pandas as pd

OUT_DIR = os.path.dirname(__file__)

SCENARIO_CUSTOMERS = pd.DataFrame([
    {"customer_id": "SCEN_CUST_HV", "account_age_days": 400, "previous_successes": 10,
     "previous_failures": 1, "avg_payment_amount": 5000, "consent_status": True,
     "preferred_payment_method": "NETBANKING"},
    {"customer_id": "SCEN_CUST_NC", "account_age_days": 60, "previous_successes": 2,
     "previous_failures": 1, "avg_payment_amount": 1500, "consent_status": False,
     "preferred_payment_method": "UPI"},
    {"customer_id": "SCEN_CUST_RL", "account_age_days": 200, "previous_successes": 5,
     "previous_failures": 2, "avg_payment_amount": 1200, "consent_status": True,
     "preferred_payment_method": "CARD"},
])

SCENARIO_PAYMENTS = pd.DataFrame([
    {
        # REAL threshold is Rs 50,000 (backend/policy/policy_engine.py MAX_ACTION_AMOUNT)
        # -- not weakened for this test.
        "payment_id": "SCEN_HIGHVALUE", "customer_id": "SCEN_CUST_HV", "amount": 75000.00,
        "payment_method": "NETBANKING", "status": "FAILED", "failure_reason": "BANK_TIMEOUT",
        "retry_count": 0,
    },
    {
        # EXPIRED_CARD strongly favors PAYMENT_LINK on pure expected value -- a real test
        # of whether the policy engine, not the agent, is what catches the violation.
        "payment_id": "SCEN_NOCONSENT", "customer_id": "SCEN_CUST_NC", "amount": 2200.00,
        "payment_method": "NETBANKING", "status": "FAILED", "failure_reason": "EXPIRED_CARD",
        "retry_count": 0,
    },
    {
        # BANK_TIMEOUT has a naturally high retry-success probability, so even at
        # retry_count=3 the agent's own economics will likely still favor a retry --
        # this is what actually forces the max_retries policy rule to intervene,
        # rather than the agent avoiding retries on its own.
        "payment_id": "SCEN_MAXRETRY", "customer_id": "SCEN_CUST_RL", "amount": 1200.00,
        "payment_method": "CARD", "status": "FAILED", "failure_reason": "BANK_TIMEOUT",
        "retry_count": 3,
    },
])

if __name__ == "__main__":
    SCENARIO_CUSTOMERS.to_csv(os.path.join(OUT_DIR, "scenario_customers.csv"), index=False)
    SCENARIO_PAYMENTS.to_csv(os.path.join(OUT_DIR, "scenario_payments.csv"), index=False)
    print("Wrote scenario_customers.csv and scenario_payments.csv")
    print("Scenario IDs: SCEN_HIGHVALUE, SCEN_NOCONSENT, SCEN_MAXRETRY")
    print("These never touch the original dataset files.")