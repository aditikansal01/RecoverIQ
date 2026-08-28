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

import json
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

import tools

load_dotenv()

MODEL = "gemini-flash-lite-latest"  # confirmed working on this key

SYSTEM_PROMPT = """You are RecoverIQ's payment recovery investigation agent.

Given a failed payment_id, investigate it thoroughly before recommending an action:
1. Call get_payment_details to see the payment.
2. Call get_customer_history to see the customer's history.
3. Call calculate_expected_value for at least 3 plausible candidate actions
   (RETRY_NOW, RETRY_LATER, PAYMENT_LINK, REMINDER, ESCALATE) -- not just one.
4. Recommend the action with the best expected value, UNLESS a lower-expected-value
   action is clearly better for a reason you can state explicitly (e.g. customer
   opted out of communication, retry limit already reached).

All monetary amounts are in Indian Rupees -- always write "Rs" before amounts in your
reasoning, never a dollar sign.

You NEVER invent a probability or expected value yourself -- always call the tools.

Recommend based on expected value alone. Do NOT factor in customer consent, contact
preferences, retry limits, or any other compliance/policy constraint when choosing your
recommendation -- that enforcement happens in a separate, deterministic policy layer
downstream, on purpose, so compliance rules stay auditable and can't be silently
reasoned around by the model. Just tell me what maximizes expected recovery.

When you have enough evidence, respond with ONLY a JSON object (no markdown fences,
no extra text) in this exact shape:
{
  "payment_id": "...",
  "recommended_action": "...",
  "expected_value": <number>,
  "confidence": "low" | "medium" | "high",
  "reasoning": "2-4 sentences explaining why, referencing the actual numbers you found"
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
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            text = (response.text or "").strip()
            if verbose:
                print(f"\n--- Final model output (turn {turn}) ---\n{text}\n")
            try:
                cleaned = text.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
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