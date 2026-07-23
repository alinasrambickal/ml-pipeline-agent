"""
agents/evaluator.py

Evaluator agent — reads stdout, stderr, and metrics from a training run and
proposes what hyperparameters to try next. It is only ever called after
loop.py has already deterministically decided (via stopping.py) that this
branch is continuing, so it doesn't decide should_continue — that's not a
judgment call, it's a fact code can check directly. Each branch tunes one
fixed model (see loop.py's parallel per-model branches), so the Evaluator
no longer proposes next_model either — there's nothing to switch to within
a branch. Its proposed next_params still gets validated by the caller
(registry.py) before being trusted; this function itself does no
business-rule enforcement, only the LLM call and response parsing.
"""

import json
from config import GROQ_MODEL, AGENT_TEMPERATURE
from groq_client import call_groq
from registry import get_model_prompt_block

SYSTEM_PROMPT = """You are an ML evaluator reviewing the results of a training run.

You are tuning ONLY {model_name} for this entire run — there is no other
model to switch to. Your job is to diagnose this run and propose different
parameter values to try next for the same model. The loop has already
determined it will run another iteration; do not decide whether to continue.

Output a JSON decision with exactly these keys:
  - reason: one sentence diagnosing this run's result
  - next_strategy: plain-English description of what to try next
  - next_params: a dict with ALL valid params for {model_name}, fully specified

Rules:
  - If stderr contains a Python traceback, diagnose it and set next_strategy to exactly how to fix it.
  - If metrics are empty or missing, treat it as a failed run — still set next_params.
  - If previous metrics are 'none' because the previous iteration crashed (not because this is truly iteration 1), treat this as iteration 1 for improvement purposes.
  - A ConvergenceWarning in stderr is NOT a failure — metrics are still valid. Increase max_iter or add StandardScaler to fix it.
  - A ConvergenceWarning is only worth fixing ONCE. If the previous iteration already attempted to fix convergence and improvement is still 0.0, try a different kind of change (not another max_iter increase).
  - You have access to the full experiment history below. Never suggest a params combination that was already tried in any previous iteration, not just the last one.
  - At least one param value must be meaningfully different from every previously tried combination.

CRITICAL — next_params:
  - next_params MUST include every param listed under valid_params below — no omissions, no extras.
  - For LogisticRegression, solver+penalty MUST be one of the listed solver_penalty_combinations exactly.

{registry_block}

Output ONLY valid JSON. No explanation, no markdown fences."""


def run_evaluator(
    model_name: str,
    metrics: dict,
    stdout: str,
    stderr: str,
    iteration: int,
    iterations_per_branch: int,
    previous_metrics: dict | None = None,
    history: list[dict] | None = None,
    primary_metric: str = "accuracy",
    rejection_reason: str | None = None,
) -> dict:
    """
    Call the Evaluator agent and return a proposal dict.

    Args:
        rejection_reason: set on retry — the caller's specific reason the
            previous proposal was rejected, fed back so the LLM can self-correct.

    Returns:
        Dict with keys: reason, next_strategy, next_params.
    """

    rendered_prompt = SYSTEM_PROMPT.format(
        model_name=model_name,
        registry_block=get_model_prompt_block(model_name),
    )

    rejection_block = ""
    if rejection_reason:
        rejection_block = f"Your previous proposal was rejected — fix it:\n{rejection_reason}\n"

    user_message = (
        f"Iteration: {iteration} of {iterations_per_branch}\n"
        f"Primary metric: {primary_metric}\n"
        f"Metrics this run: {json.dumps(metrics)}\n"
        f"Previous metrics: {json.dumps(previous_metrics) if previous_metrics else 'none (either first iteration or previous iteration crashed)'}\n"
        f"stdout (truncated to 1000 chars):\n{stdout[:1000]}\n"
        f"stderr (truncated to 1000 chars):\n{stderr[-1000:]}\n"
        f"{_format_history(history)}"
        f"{rejection_block}"
    )

    response = call_groq(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=AGENT_TEMPERATURE,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if LLM ignored instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt repair: use LLM to fix its own malformed JSON
        repair_response = call_groq(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": f"Fix this invalid JSON so it parses correctly. Output ONLY the fixed JSON, nothing else:\n{raw}"}
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = repair_response.choices[0].message.content.strip()
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Evaluator returned invalid JSON after repair attempt: {e}\nRaw:\n{raw}")

    _validate_decision(decision)
    return decision


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "Experiment history: none (this is the first iteration).\n"
    lines = ["Experiment history (ALL prior iterations this run):"]
    for r in history:
        lines.append(
            f"  Iteration {r['iteration']}: "
            f"params={r.get('params_used', {})}, "
            f"metrics={r.get('metrics', {})}, "
            f"failed={bool(r.get('error'))}"
        )
    return "\n".join(lines) + "\n"


def _validate_decision(decision: dict) -> None:
    required = {"reason", "next_strategy", "next_params"}
    missing = required - decision.keys()
    if missing:
        raise ValueError(f"Evaluator decision missing keys: {missing}")
    if not isinstance(decision["next_params"], dict):
        raise ValueError("next_params must be a dict")
