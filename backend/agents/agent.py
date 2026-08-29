"""
RecoverIQ investigation agent (v2 -- standard generate_content + automatic function
calling, not the newer Interactions API, for broader model/key compatibility).

Given a payment_id, the agent investigates using real tools (never guesses data or
probabilities), reasons over the evidence, and produces a structured recommendation.
The agent RECOMMENDS -- it never executes a money action itself. That's the policy
engine's job (Day 5).

Usage:
    python agent.py PAYMENT_ID
"""

import collections
import json
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

import tools

load_dotenv()

MODEL = "gemini-3.5-flash-lite"  # keep in sync with whatever model you're actually using

MAX_RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BACKOFF_SECONDS = 25  # doubles each retry

TARGET_RPM = 13  # stay a little under the 15/min free-tier cap
_call_timestamps = collections.deque()


def _pace_calls(verbose=True):
    """Proactively wait just long enough to stay under the rate limit, based on how
    many calls actually happened in the last 60 seconds -- avoids both wasted idle
    time and unnecessary 429s."""
    now = time.time()
    while _call_timestamps and now - _call_timestamps[0] > 60:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= TARGET_RPM:
        wait = 60 - (now - _call_timestamps[0]) + 0.5
        if wait > 0:
            if verbose:
                print(f"    [pacing: {wait:.1f}s to stay under rate limit]")
            time.sleep(wait)
    _call_timestamps.append(time.time())


def _generate_with_retry(client, model, contents, config, verbose=True):
    """Wraps generate_content with proactive pacing plus automatic backoff on any
    429s that slip through anyway."""
    delay = RATE_LIMIT_BACKOFF_SECONDS
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _pace_calls(verbose=verbose)
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if not is_rate_limit or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            if verbose:
                print(f"    [rate limited, waiting {delay}s before retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}...]")
            time.sleep(delay)
            delay *= 2


def _extract_json(text: str) -> dict | None:
    """Robustly pull a JSON object out of the model's final answer, even if it added
    stray text, a bare 'js' token, or code fences around it."""
    text = text.strip()
    # try straightforward parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first {...last matching } in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None

SYSTEM_PROMPT = """You are RecoverIQ's payment recovery investigation agent.

Given a failed payment_id, investigate it thoroughly before recommending an action:
1. Call get_payment_details to see the payment.
2. Call get_customer_history to see the customer's history.
3. Call calculate_expected_value for ALL FIVE candidate actions, every time, with
   no exceptions: RETRY_NOW, RETRY_LATER, PAYMENT_LINK, REMINDER, and ESCALATE.
   Do not skip any of them, even if one seems obviously worse -- you must see the
   actual number before ruling it out.
4. Recommend the action with the best expected value, UNLESS a lower-expected-value
   action is clearly better for a reason you can state explicitly (e.g. customer
   opted out of communication, retry limit already reached).

All monetary amounts are in Indian Rupees -- always write "Rs" before amounts in your
reasoning, never a dollar sign.

You NEVER invent a probability or expected value yourself -- always call the tools.
You NEVER decide policy limits (retry caps, consent enforcement) -- that is a separate
system's job; just make your best recommendation and note anything a human reviewer
should know.

When you have enough evidence, respond with ONLY a JSON object (no markdown fences,
no extra text) in this exact shape. Include ALL FIVE candidates you evaluated, not
just your top pick -- a separate deterministic system uses this full list to make the
final decision based on the merchant's chosen priorities:
{
  "payment_id": "...",
  "recommended_action": "...",
  "expected_value": <number>,
  "confidence": "low" | "medium" | "high",
  "reasoning": "2-4 sentences explaining why, referencing the actual numbers you found",
  "candidates": [
    {"action": "RETRY_NOW", "expected_value": <number>, "friction_score": <number>, "amount": <number>},
    {"action": "RETRY_LATER", "expected_value": <number>, "friction_score": <number>, "amount": <number>},
    {"action": "PAYMENT_LINK", "expected_value": <number>, "friction_score": <number>, "amount": <number>},
    {"action": "REMINDER", "expected_value": <number>, "friction_score": <number>, "amount": <number>},
    {"action": "ESCALATE", "expected_value": <number>, "friction_score": <number>, "amount": <number>}
  ]
}
"""

TOOL_FUNCTIONS = {
    "get_payment_details": tools.get_payment_details,
    "get_customer_history": tools.get_customer_history,
    "predict_recovery_probability": tools.predict_recovery_probability,
    "calculate_expected_value": tools.calculate_expected_value,
}


def run_agent(payment_id: str, verbose: bool = True,
              excluded_actions: list | None = None,
              rejection_reason: str | None = None) -> dict:
    client = genai.Client()  # reads GEMINI_API_KEY from env

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[
            tools.get_payment_details,
            tools.get_customer_history,
            tools.predict_recovery_probability,
            tools.calculate_expected_value,
        ],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    user_message = f"Investigate payment {payment_id} and recommend a recovery action."
    if excluded_actions:
        user_message += (
            f"\n\nIMPORTANT: The following action(s) were already recommended and REJECTED "
            f"by the policy engine: {excluded_actions}. Reason for the most recent rejection: "
            f"\"{rejection_reason}\". Do not recommend any of these actions again -- "
            f"investigate and recommend a different, still-viable action."
        )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )
    ]

    max_turns = 8
    for turn in range(1, max_turns + 1):
        response = _generate_with_retry(client, MODEL, contents, config, verbose=verbose)
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            text = (response.text or "").strip()
            if verbose:
                print(f"\n--- Final model output (turn {turn}) ---\n{text}\n")

            if not text:
                finish_reason = getattr(candidate, "finish_reason", "unknown")
                if verbose:
                    print(f"    [empty completion, finish_reason={finish_reason}]")
                if turn < max_turns:
                    # transient with lite models under function-calling load -- nudge
                    # the model to actually answer and try again rather than giving up
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text="You returned an empty response. Please provide "
                                                "the final JSON recommendation now.")],
                    ))
                    continue
                return {"error": "Model did not return valid JSON", "raw_output": text,
                        "finish_reason": str(finish_reason)}

            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
            return {"error": "Model did not return valid JSON", "raw_output": text}

        response_parts = []
        for fc in function_calls:
            fn = TOOL_FUNCTIONS.get(fc.name)
            args = dict(fc.args) if fc.args else {}
            if fn is None:
                result = {"error": f"Unknown tool {fc.name}"}
            else:
                if verbose:
                    print(f"[turn {turn}] agent calls {fc.name}({args})")
                result = fn(**args)
                if verbose:
                    print(f"    -> {result}")
            response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return {"error": f"Agent did not converge to a final answer in {max_turns} turns"}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python agent.py PAYMENT_ID")
        sys.exit(1)

    payment_id = sys.argv[1]
    result = run_agent(payment_id)
    print("\n=== RECOMMENDATION ===")
    print(json.dumps(result, indent=2))