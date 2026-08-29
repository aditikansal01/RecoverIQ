"""
Direct unit tests for the policy engine. These test each rule in isolation by
calling evaluate_policy() with constructed inputs -- deterministic, zero API cost,
and the correct way to prove a rule works even in cases (like max_retries) that
the agent's own economics may rarely trigger on its own.

Usage:
    python test_policy_engine.py
    (or: pytest test_policy_engine.py -v, if pytest is installed)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend", "policy"))
from policy_engine import evaluate_policy, MAX_RETRIES, MAX_ACTION_AMOUNT  # noqa: E402


def test_normal_payment_is_allowed():
    payment = {"amount": 4999, "retry_count": 0}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "RETRY_LATER")
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result}"


def test_retry_blocked_at_max_retries():
    """Direct test of the max_retries rule -- proven here even though the agent's
    own economics may rarely recommend a retry at this point (see README)."""
    payment = {"amount": 1200, "retry_count": MAX_RETRIES}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "RETRY_NOW")
    assert result["decision"] == "BLOCK", f"Expected BLOCK, got {result}"
    assert result["rule"] == "max_retries"

    result2 = evaluate_policy(payment, customer, "RETRY_LATER")
    assert result2["decision"] == "BLOCK", f"Expected BLOCK, got {result2}"
    assert result2["rule"] == "max_retries"


def test_retry_allowed_below_max_retries():
    payment = {"amount": 1200, "retry_count": MAX_RETRIES - 1}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "RETRY_NOW")
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result}"


def test_high_value_requires_review():
    payment = {"amount": MAX_ACTION_AMOUNT + 1, "retry_count": 0}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "ESCALATE")
    assert result["decision"] == "REVIEW", f"Expected REVIEW, got {result}"
    assert result["rule"] == "high_value_review"


def test_high_value_review_applies_regardless_of_action():
    """The rule should trigger for ANY action once the amount exceeds the threshold,
    not just specific ones -- this is what 'gated' means in the project's pitch."""
    payment = {"amount": MAX_ACTION_AMOUNT + 1, "retry_count": 0}
    customer = {"consent_status": True}
    for action in ["RETRY_NOW", "RETRY_LATER", "PAYMENT_LINK", "REMINDER", "ESCALATE"]:
        result = evaluate_policy(payment, customer, action)
        assert result["decision"] == "REVIEW", f"Action {action} should trigger REVIEW, got {result}"


def test_communication_blocked_without_consent():
    payment = {"amount": 2000, "retry_count": 0}
    customer = {"consent_status": False}
    for action in ["PAYMENT_LINK", "REMINDER"]:
        result = evaluate_policy(payment, customer, action)
        assert result["decision"] == "BLOCK", f"Expected BLOCK for {action}, got {result}"
        assert result["rule"] == "consent_required"


def test_retry_allowed_without_consent():
    """No-consent should only block communication actions, not silent retries --
    this is the important negative case, since a blanket block would be wrong."""
    payment = {"amount": 2000, "retry_count": 0}
    customer = {"consent_status": False}
    result = evaluate_policy(payment, customer, "RETRY_NOW")
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result}"


def test_duplicate_action_blocked():
    payment = {"amount": 2000, "retry_count": 0}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "PAYMENT_LINK", previously_tried_actions=["PAYMENT_LINK"])
    assert result["decision"] == "BLOCK", f"Expected BLOCK, got {result}"
    assert result["rule"] == "no_duplicate_action"


def test_non_duplicate_action_allowed():
    payment = {"amount": 2000, "retry_count": 0}
    customer = {"consent_status": True}
    result = evaluate_policy(payment, customer, "REMINDER", previously_tried_actions=["PAYMENT_LINK"])
    assert result["decision"] == "ALLOW", f"Expected ALLOW, got {result}"


ALL_TESTS = [
    test_normal_payment_is_allowed,
    test_retry_blocked_at_max_retries,
    test_retry_allowed_below_max_retries,
    test_high_value_requires_review,
    test_high_value_review_applies_regardless_of_action,
    test_communication_blocked_without_consent,
    test_retry_allowed_without_consent,
    test_duplicate_action_blocked,
    test_non_duplicate_action_allowed,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    sys.exit(1 if failed else 0)