"""
Deterministic policy engine. The agent recommends; this decides whether that
recommendation is allowed to execute. No LLM involvement here on purpose --
every rule below is auditable, testable, and explainable without needing to
ask the model "why."

Decision is one of: ALLOW, REVIEW, BLOCK.
"""

MAX_RETRIES = 3
MAX_ACTION_AMOUNT = 50000  # above this, require human review regardless of action
COMMUNICATION_ACTIONS = {"PAYMENT_LINK", "REMINDER"}
RETRY_ACTIONS = {"RETRY_NOW", "RETRY_LATER"}


def evaluate_policy(payment: dict, customer: dict, action: str,
                     previously_tried_actions: list[str] | None = None) -> dict:
    """
    payment: dict from tools.get_payment_details (needs amount, retry_count)
    customer: dict from tools.get_customer_history (needs consent_status)
    action: the agent's recommended action
    previously_tried_actions: actions already rejected earlier in this same
        investigation (used to block the agent recommending the same thing twice)

    Returns: {"decision": "ALLOW"|"REVIEW"|"BLOCK", "reason": str, "rule": str}
    """
    previously_tried_actions = previously_tried_actions or []

    # Rule 1: retry limit
    if action in RETRY_ACTIONS and payment["retry_count"] >= MAX_RETRIES:
        return {
            "decision": "BLOCK",
            "reason": f"Maximum retry limit ({MAX_RETRIES}) already reached for this payment. "
                      f"No further retries permitted -- consider escalation instead.",
            "rule": "max_retries",
        }

    # Rule 2: high-value transactions need human review, regardless of action
    if payment["amount"] > MAX_ACTION_AMOUNT:
        return {
            "decision": "REVIEW",
            "reason": f"Amount Rs {payment['amount']:,.2f} exceeds the Rs {MAX_ACTION_AMOUNT:,} "
                      f"auto-execution limit. Routing to human review.",
            "rule": "high_value_review",
        }

    # Rule 3: consent -- never contact a customer who opted out
    if action in COMMUNICATION_ACTIONS and not customer.get("consent_status", True):
        return {
            "decision": "BLOCK",
            "reason": "Customer has not consented to communication. Cannot send a "
                      "payment link or reminder -- consider a silent retry instead.",
            "rule": "consent_required",
        }

    # Rule 4: don't repeat an action already tried and rejected/failed in this investigation
    if action in previously_tried_actions:
        return {
            "decision": "BLOCK",
            "reason": f"Action {action} was already attempted in this investigation. "
                      f"Recommend a different action.",
            "rule": "no_duplicate_action",
        }

    return {
        "decision": "ALLOW",
        "reason": "Passed all policy checks.",
        "rule": "none",
    }