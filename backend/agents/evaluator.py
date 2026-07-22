"""
agents/evaluator.py
 
Evaluator agent — reads stdout, stderr, and metrics from a training run and
decides whether to continue iterating and what to try next.
"""
 
import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, AGENT_TEMPERATURE
from registry import get_registry_prompt_block, enforce_model_switch
 
# _client = Groq(api_key=GROQ_API_KEY) //commented out for lazy instantiation
 
SYSTEM_PROMPT = """You are an ML evaluator reviewing the results of a training run.
 
Output a JSON decision with exactly these keys:
  - should_continue: true | false
  - reason: one sentence explaining your decision
  - next_strategy: plain-English description of what to try next (empty string if not continuing)
  - next_model: the model class name to use next (empty string if not continuing)
  - next_params: a dict with ALL valid params for next_model, fully specified (empty dict if not continuing)
  - improvement: numeric improvement in the primary metric vs last iteration, or null if first iteration
 
Rules:
  - If stderr contains a Python traceback, diagnose it and set next_strategy to exactly how to fix it.
  - If metrics are empty or missing, treat it as a failed run — still set next_model and next_params.
  - Stop (should_continue: false) if: max iterations reached, metric > 0.97, or two consecutive iterations show no improvement.
  - If previous metrics are 'none' because the previous iteration crashed (not because this is truly iteration 1), treat this as iteration 1 for improvement purposes — do not penalize by switching models prematurely.
  - A ConvergenceWarning in stderr is NOT a failure — metrics are still valid. Increase max_iter or add StandardScaler to fix it.
  - A ConvergenceWarning is only worth fixing ONCE. If the previous iteration already attempted to fix convergence (increased max_iter or added scaling) and improvement is still 0.0, you MUST switch to a different model class — do not suggest further max_iter increases.
  - You have access to the full experiment history below. Never suggest a model+params combination that was already tried in any previous iteration, not just the last one.
  - If all registry models have been tried and none improved, set should_continue: false.
  - Never suggest next_params that are identical to the params used in any previous iteration. If you are keeping the same model class, at least one param value must be meaningfully different.
 
CRITICAL — next_model and next_params:
  - next_model MUST be one of the models listed in the registry below.
  - next_params MUST include every param listed under valid_params for that model — no omissions, no extras.
  - For LogisticRegression, solver+penalty MUST be one of the listed solver_penalty_combinations exactly.
  - If improvement >= 0.01, you may keep the same model class but must change at least one param value.
  - If improvement < 0.01, switch to a different model class than what was just used.
 
{registry_block}
 
Output ONLY valid JSON. No explanation, no markdown fences."""
 
 
def run_evaluator(
    metrics: dict,
    stdout: str,
    stderr: str,
    iteration: int,
    max_iterations: int,
    model_used: str,
    previous_metrics: dict | None = None,
    history: list[dict] | None = None,
    primary_metric: str = "accuracy",
) -> dict:
    """
    Call the Evaluator agent and return a validated decision dict.
 
    Returns:
        Dict with keys: should_continue, reason, next_strategy, next_model, next_params, improvement.
    """
 
    client = Groq(api_key=GROQ_API_KEY)
 
    rendered_prompt = SYSTEM_PROMPT.format(registry_block=get_registry_prompt_block())
 
    user_message = (
        f"Iteration: {iteration} of {max_iterations}\n"
        f"Primary metric: {primary_metric}\n"
        f"Metrics this run: {json.dumps(metrics)}\n"
        f"Previous metrics: {json.dumps(previous_metrics) if previous_metrics else 'none (either first iteration or previous iteration crashed)'}\n"
        f"stdout (truncated to 1000 chars):\n{stdout[:1000]}\n"
        f"stderr (truncated to 1000 chars):\n{stderr[-1000:]}\n"
        f"{_format_history(history)}"
    )
 
    response = client.chat.completions.create(
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
        repair_response = client.chat.completions.create(
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

    # Hard override: force a model switch if the LLM ignored the no-improvement rule
    decision = enforce_model_switch(decision, model_used, metrics, previous_metrics, history, primary_metric)

    # Hard override: never continue past max_iterations regardless of LLM output
    if iteration >= max_iterations:
        decision["should_continue"] = False
        decision["reason"] = f"Reached max iterations ({max_iterations})."
 
    return decision
 
 
def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "Experiment history: none (this is the first iteration).\n"
    lines = ["Experiment history (ALL prior iterations this run):"]
    for r in history:
        lines.append(
            f"  Iteration {r['iteration']}: "
            f"model={r.get('model_used', 'unknown')}, "
            f"params={r.get('params_used', {})}, "
            f"metrics={r.get('metrics', {})}, "
            f"failed={bool(r.get('error'))}"
        )
    return "\n".join(lines) + "\n"
 
 
def _validate_decision(decision: dict) -> None:
    required = {"should_continue", "reason", "next_strategy", "next_model", "next_params", "improvement"}
    missing = required - decision.keys()
    if missing:
        raise ValueError(f"Evaluator decision missing keys: {missing}")
    if not isinstance(decision["should_continue"], bool):
        raise ValueError("should_continue must be a boolean")
    