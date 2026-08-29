"""
Deterministic decision engine for merchant-configurable optimization modes.

This is intentionally NOT part of the LLM's job -- same principle as the policy
engine. The agent investigates and computes evidence (expected value + friction for
every candidate action); this module decides which candidate wins, based on a
merchant-chosen mode. Changing modes never requires touching the agent's prompt.

The friction penalty is a PERCENTAGE OF THE PAYMENT AMOUNT per friction point, not a
flat rupee amount -- a flat penalty is meaningless on a Rs 75,000 payment and
overwhelming on a Rs 150 one, so it has to scale with the transaction.

MAX_RECOVERY : friction ignored, pure expected-value maximization (backward-compatible
               with the original agent behavior -- this is the mode used for the
               100-payment evaluation reported in the README).
BALANCED     : expected value discounted by a moderate percentage of the payment
               amount per friction point.
MIN_FRICTION : expected value discounted heavily -- only picks a higher-friction
               action if its EV advantage clearly justifies the extra customer contact.

Note: on some payments, BALANCED and MIN_FRICTION can select the same action as each
other (or even as MAX_RECOVERY). This happens when one action's EV advantage per
friction point is genuinely larger than any reasonable penalty would offset -- that's
a real property of the specific payment's numbers, not a bug in the scoring. The modes
are honestly different formulas; they don't guarantee three different answers on every
single payment.
"""

FRICTION_PENALTY_PCT = {
    "MAX_RECOVERY": 0.0,
    "BALANCED": 0.02,     # 2% of payment amount per friction point
    "MIN_FRICTION": 0.12,  # 12% of payment amount per friction point
}

VALID_MODES = set(FRICTION_PENALTY_PCT.keys())


def score_candidate(candidate: dict, mode: str) -> float:
    """candidate needs: expected_value, friction_score, amount"""
    pct = FRICTION_PENALTY_PCT[mode]
    penalty = pct * candidate["friction_score"] * candidate["amount"]
    return candidate["expected_value"] - penalty


def select_action(candidates: list[dict], mode: str = "MAX_RECOVERY") -> dict:
    """
    candidates: list of dicts, each with at least
        {"action": str, "expected_value": float, "friction_score": int}
    mode: one of MAX_RECOVERY, BALANCED, MIN_FRICTION

    Returns the winning candidate dict, with a "score" key added showing how it was
    chosen, and "mode" recorded for the audit trail.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
    if not candidates:
        raise ValueError("No candidates provided")

    scored = [
        {**c, "score": round(score_candidate(c, mode), 2), "mode": mode}
        for c in candidates
    ]
    winner = max(scored, key=lambda c: c["score"])
    return winner