"""
PaymentSimulator -- stands in for real payment execution.

This module holds the "true" recovery-probability model (BASE_PROB / recovery_prob).
In a real system this would be reality itself -- unknown, unobservable in closed form.
Here we simulate it so we can (a) generate realistic historical data, and (b) let the
ML model (Day 3) try to *approximate* this same function purely from observed outcomes,
exactly as it would have to in production.

The agent and policy engine never get to see BASE_PROB or recovery_prob directly --
only the ML model's *estimate* of them, learned from data. Keeping the "true" model
and the "learned" model separate is what makes the evaluation honest.
"""

import random
import numpy as np

FAILURE_REASONS = [
    "BANK_TIMEOUT", "EXPIRED_CARD", "INSUFFICIENT_FUNDS",
    "NETWORK_ERROR", "MANDATE_AUTH_FAILED", "CHECKOUT_ABANDONED", "UNKNOWN",
]

ACTIONS = ["RETRY_NOW", "RETRY_LATER", "PAYMENT_LINK", "REMINDER", "ESCALATE"]

BASE_PROB = {
    "BANK_TIMEOUT":        {"RETRY_NOW": 0.35, "RETRY_LATER": 0.72, "PAYMENT_LINK": 0.55, "REMINDER": 0.30, "ESCALATE": 0.95},
    "EXPIRED_CARD":        {"RETRY_NOW": 0.04, "RETRY_LATER": 0.06, "PAYMENT_LINK": 0.68, "REMINDER": 0.40, "ESCALATE": 0.90},
    "INSUFFICIENT_FUNDS":  {"RETRY_NOW": 0.12, "RETRY_LATER": 0.48, "PAYMENT_LINK": 0.38, "REMINDER": 0.35, "ESCALATE": 0.85},
    "NETWORK_ERROR":       {"RETRY_NOW": 0.61, "RETRY_LATER": 0.58, "PAYMENT_LINK": 0.45, "REMINDER": 0.25, "ESCALATE": 0.92},
    "MANDATE_AUTH_FAILED": {"RETRY_NOW": 0.05, "RETRY_LATER": 0.10, "PAYMENT_LINK": 0.60, "REMINDER": 0.30, "ESCALATE": 0.88},
    "CHECKOUT_ABANDONED":  {"RETRY_NOW": 0.02, "RETRY_LATER": 0.05, "PAYMENT_LINK": 0.42, "REMINDER": 0.51, "ESCALATE": 0.60},
    "UNKNOWN":             {"RETRY_NOW": 0.20, "RETRY_LATER": 0.28, "PAYMENT_LINK": 0.30, "REMINDER": 0.22, "ESCALATE": 0.75},
}

# Action costs/friction used later by the expected-value engine (Day 3+).
# cost = rough operational cost in Rs, friction = 0 (none) to 3 (high, e.g. repeated contact)
ACTION_COST = {
    "RETRY_NOW":    {"cost": 2,  "friction": 0},
    "RETRY_LATER":  {"cost": 2,  "friction": 0},
    "PAYMENT_LINK": {"cost": 5,  "friction": 1},
    "REMINDER":     {"cost": 3,  "friction": 1},
    "ESCALATE":     {"cost": 250, "friction": 3},
}


def recovery_prob(failure_reason, action, retry_count, previous_successes, previous_failures, account_age_days):
    p = BASE_PROB[failure_reason][action]
    p *= max(0.25, 1 - 0.22 * retry_count)
    history_score = previous_successes - previous_failures
    p *= np.clip(1 + 0.015 * history_score, 0.6, 1.25)
    if account_age_days < 14:
        p *= 0.85
    return float(np.clip(p, 0.01, 0.98))


def baseline_action(failure_reason: str, retry_count: int) -> str:
    """Deterministic, non-AI baseline strategy -- our control group."""
    if retry_count >= 3:
        return "ESCALATE"
    return {
        "BANK_TIMEOUT": "RETRY_LATER",
        "EXPIRED_CARD": "PAYMENT_LINK",
        "INSUFFICIENT_FUNDS": "RETRY_LATER",
        "NETWORK_ERROR": "RETRY_NOW",
        "MANDATE_AUTH_FAILED": "PAYMENT_LINK",
        "CHECKOUT_ABANDONED": "REMINDER",
        "UNKNOWN": "REMINDER",
    }[failure_reason]


class PaymentSimulator:
    """Executes a recovery action against a payment and returns a realistic outcome.

    Occasionally injects a technical failure (TIMEOUT) independent of the underlying
    recovery probability -- this is what Day 5's failure-handling / replanning logic
    reacts to, separate from a normal "customer didn't pay" outcome.
    """

    def __init__(self, technical_failure_rate: float = 0.04, seed: int | None = None):
        self.technical_failure_rate = technical_failure_rate
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def execute_action(self, payment: dict, action: str) -> dict:
        """
        payment: dict with keys failure_reason, retry_count, previous_successes,
                 previous_failures, account_age_days, amount
        action: one of ACTIONS
        returns: dict with execution_status (SUCCESS/FAILED/TIMEOUT), recovered (bool),
                 recovered_amount (float)
        """
        # Simulate an infrastructure-level failure (API timeout etc.) independent of
        # whether the action *would* have worked -- this is the failure case we
        # deliberately show being handled gracefully in the demo.
        if self._rng.random() < self.technical_failure_rate:
            return {
                "execution_status": "TIMEOUT",
                "recovered": False,
                "recovered_amount": 0.0,
            }

        p = recovery_prob(
            payment["failure_reason"], action, payment["retry_count"],
            payment["previous_successes"], payment["previous_failures"],
            payment["account_age_days"],
        )
        recovered = self._rng.random() < p
        return {
            "execution_status": "SUCCESS" if recovered else "FAILED",
            "recovered": recovered,
            "recovered_amount": float(payment["amount"]) if recovered else 0.0,
        }