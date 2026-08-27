"""
Tools available to the recovery agent. Each function here is what the LLM actually
calls -- it never invents payment data or probabilities itself, only reasons over
what these functions return.
"""

import os
import sys

import pandas as pd

_ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
sys.path.insert(0, _ML_DIR)
import predict as _predict_module  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
from payment_simulator import ACTION_COST, ACTIONS  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")

_payments = None
_customers = None


def _load_data():
    global _payments, _customers
    if _payments is None:
        hist = pd.read_csv(os.path.join(_DATA_DIR, "payments.csv"))
        ev = pd.read_csv(os.path.join(_DATA_DIR, "payments_eval.csv"))
        _payments = pd.concat([hist, ev], ignore_index=True).set_index("payment_id")
        _customers = pd.read_csv(os.path.join(_DATA_DIR, "customers.csv")).set_index("customer_id")


def get_payment_details(payment_id: str) -> dict:
    """Look up a payment's amount, failure reason, retry count, and status."""
    _load_data()
    if payment_id not in _payments.index:
        return {"error": f"No payment found with id {payment_id}"}
    row = _payments.loc[payment_id]
    return {
        "payment_id": payment_id,
        "customer_id": row["customer_id"],
        "amount": float(row["amount"]),
        "payment_method": row["payment_method"],
        "failure_reason": row["failure_reason"],
        "retry_count": int(row["retry_count"]),
        "status": row["status"],
    }


def get_customer_history(customer_id: str) -> dict:
    """Look up a customer's account age, payment history, and consent status."""
    _load_data()
    if customer_id not in _customers.index:
        return {"error": f"No customer found with id {customer_id}"}
    row = _customers.loc[customer_id]
    return {
        "customer_id": customer_id,
        "account_age_days": int(row["account_age_days"]),
        "previous_successes": int(row["previous_successes"]),
        "previous_failures": int(row["previous_failures"]),
        "consent_status": bool(row["consent_status"]),
        "preferred_payment_method": row["preferred_payment_method"],
    }


def predict_recovery_probability(payment_id: str, action: str) -> dict:
    """Get the ML model's estimated probability that a given action will recover
    this payment. This is a real, data-backed number -- never guess this yourself."""
    if action not in ACTIONS:
        return {"error": f"Unknown action '{action}'. Valid actions: {ACTIONS}"}
    payment = get_payment_details(payment_id)
    if "error" in payment:
        return payment
    customer = get_customer_history(payment["customer_id"])
    if "error" in customer:
        return customer

    prob = _predict_module.predict_recovery_probability(
        failure_reason=payment["failure_reason"],
        action=action,
        retry_count=payment["retry_count"],
        previous_successes=customer["previous_successes"],
        previous_failures=customer["previous_failures"],
        account_age_days=customer["account_age_days"],
        amount=payment["amount"],
    )
    return {"payment_id": payment_id, "action": action, "recovery_probability": round(prob, 4)}


def calculate_expected_value(payment_id: str, action: str) -> dict:
    """Calculate expected recovered value for an action, net of its operational cost.
    expected_value = P(recovery) x amount - action_cost. Also returns a friction score
    (0=none to 3=high) representing customer-experience cost of this action."""
    if action not in ACTION_COST:
        return {"error": f"Unknown action '{action}'"}
    payment = get_payment_details(payment_id)
    if "error" in payment:
        return payment

    prob_result = predict_recovery_probability(payment_id, action)
    if "error" in prob_result:
        return prob_result
    prob = prob_result["recovery_probability"]

    cost = ACTION_COST[action]["cost"]
    friction = ACTION_COST[action]["friction"]
    expected_value = round(prob * payment["amount"] - cost, 2)

    return {
        "payment_id": payment_id,
        "action": action,
        "recovery_probability": prob,
        "amount": payment["amount"],
        "action_cost": cost,
        "friction_score": friction,
        "expected_value": expected_value,
    }