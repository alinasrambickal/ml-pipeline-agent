"""
agents/planner.py

Planner agent — takes a dataset description + task and returns a structured
JSON plan. No CSV data is sent to the LLM; only column names and row count
are shared to avoid leaking dataset contents.
"""

import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, AGENT_TEMPERATURE

# Client is instantiated here (backend only). Key never leaves this process.
# _client = Groq(api_key=GROQ_API_KEY) //commented out for lazy instantiation

SYSTEM_PROMPT = """You are an ML planning agent.

Given a dataset description (column names, row count, dtypes) and a plain-English
task, output a JSON plan with exactly these keys:
  - task_type: "classification" | "regression"
  - target_column: string (must be one of the provided columns)
  - metric: "accuracy" | "f1" | "rmse" | "mae"
  - initial_strategy: one sentence describing the first modelling approach
  - suggested_models: list of 3 sklearn estimator class names to try in order

Output ONLY valid JSON. No explanation, no markdown fences, no extra text."""


def run_planner(dataset_description: str, task_description: str) -> dict:
    """
    Call the Planner agent and return a validated plan dict.

    Args:
        dataset_description: A safe summary (columns, dtypes, row count) —
                             NOT raw CSV data.
        task_description: Plain-English task from the user.

    Returns:
        Parsed and validated plan dict.

    Raises:
        ValueError: if the LLM returns malformed JSON or missing keys.
    """

    client = Groq(api_key=GROQ_API_KEY)

    user_message = (
        f"Dataset:\n{dataset_description}\n\n"
        f"Task: {task_description}"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=AGENT_TEMPERATURE,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}\nRaw output:\n{raw}")

    _validate_plan(plan)
    return plan


def _validate_plan(plan: dict) -> None:
    from registry import MODEL_REGISTRY  # local import avoids circular deps

    required_keys = {
        "task_type", "target_column", "metric",
        "initial_strategy", "suggested_models"
    }
    missing = required_keys - plan.keys()
    if missing:
        raise ValueError(f"Planner plan missing keys: {missing}")

    if plan["task_type"] not in ("classification", "regression"):
        raise ValueError(f"Invalid task_type: {plan['task_type']}")

    valid_metrics = {"accuracy", "f1", "rmse", "mae"}
    if plan["metric"] not in valid_metrics:
        raise ValueError(f"Invalid metric: {plan['metric']}")

    if not isinstance(plan["suggested_models"], list) or not plan["suggested_models"]:
        raise ValueError("suggested_models must be a non-empty list")

    unknown_models = [m for m in plan["suggested_models"] if m not in MODEL_REGISTRY]
    if unknown_models:
        raise ValueError(f"Planner suggested models not in registry: {unknown_models}")
