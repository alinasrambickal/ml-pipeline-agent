"""
registry.py

Single source of truth for MODEL_REGISTRY logic.
All agents and loop.py import from here — never directly from config.
"""

from config import MODEL_REGISTRY, LOWER_IS_BETTER_METRICS


def get_registry_prompt_block() -> str:
    """
    Returns a formatted string describing the full registry,
    injected into both the coder and evaluator prompts.
    """
    lines = ["ALLOWED MODELS AND PARAMETERS:"]
    for model_name, spec in MODEL_REGISTRY.items():
        lines.append(f"\n  {model_name}:")
        lines.append(f"    valid_params: {spec['valid_params']}")
        lines.append(f"    defaults: {spec['defaults']}")
        if "solver_penalty_combinations" in spec:
            lines.append(f"    solver_penalty_combinations (MUST use one of these exactly):")
            for combo in spec["solver_penalty_combinations"]:
                lines.append(f"      - solver={combo['solver']}, penalty={combo['penalty']}")
    return "\n".join(lines)


def validate_next_params(model_name: str, params: dict) -> None:
    """
    Validates that:
    1. model_name is in the registry
    2. params contains exactly the keys in valid_params (no more, no less)
    3. For LogisticRegression, solver/penalty combo is in the allowed list

    Raises ValueError with a clear message on any violation.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not in MODEL_REGISTRY. Allowed: {list(MODEL_REGISTRY.keys())}")

    spec = MODEL_REGISTRY[model_name]
    valid_params = set(spec["valid_params"])
    provided_params = set(params.keys())

    missing = valid_params - provided_params
    extra = provided_params - valid_params

    if missing:
        raise ValueError(f"{model_name}: next_params missing required keys: {missing}")
    if extra:
        raise ValueError(f"{model_name}: next_params contains unknown keys: {extra}")

    if "solver_penalty_combinations" in spec:
        combo = {"solver": params.get("solver"), "penalty": params.get("penalty")}
        if combo not in spec["solver_penalty_combinations"]:
            raise ValueError(
                f"LogisticRegression: solver={combo['solver']} + penalty={combo['penalty']} "
                f"is not a valid combination. Allowed: {spec['solver_penalty_combinations']}"
            )


def get_defaults(model_name: str) -> dict:
    """Returns the default params for a model. Used as fallback on validation failure."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in registry.")
    return dict(MODEL_REGISTRY[model_name]["defaults"])


def best_metric_value(metrics: dict, primary_metric: str) -> float:
    """
    Returns a comparable float for sorting — negated for lower-is-better metrics
    so max() always picks the best model correctly.
    """
    val = metrics.get(primary_metric, None)
    if val is None:
        return float("-inf")
    return -val if primary_metric in LOWER_IS_BETTER_METRICS else val


def enforce_model_switch(
    decision: dict,
    model_used: str,
    metrics: dict,
    previous_metrics: dict | None,
    history: list[dict] | None,
    primary_metric: str,
) -> dict:
    """
    Deterministic guardrail: if the model that just ran (model_used) is the
    same one used in the previous iteration and improvement was < 0.01, force
    a switch to an untried model if the evaluator still proposes staying on it.

    The LLM doesn't reliably follow this rule from prompting alone, so it's
    enforced here in code — same principle as validate_next_params. Improvement
    is recomputed from actual metrics rather than trusting the LLM-reported
    'improvement' field. Compares against the iteration that JUST ran (passed
    in explicitly), not just entries already in `history` — history only holds
    iterations before this one, so comparing history[-1] vs history[-2] alone
    lags a full iteration behind and can never catch the pattern in time.
    """
    if not decision.get("should_continue") or not decision.get("next_model"):
        return decision
    if not history or not previous_metrics:
        return decision  # no prior completed iteration to compare against yet

    prev_model = history[-1].get("model_used")
    if not prev_model or prev_model != model_used:
        return decision

    improvement = best_metric_value(metrics, primary_metric) - \
        best_metric_value(previous_metrics, primary_metric)
    if improvement >= 0.01:
        return decision

    if decision["next_model"] != model_used:
        return decision  # evaluator already chose to switch — nothing to enforce

    tried_models = {r.get("model_used") for r in history if r.get("model_used")}
    tried_models.add(model_used)
    candidates = [m for m in MODEL_REGISTRY if m not in tried_models]
    if not candidates:
        candidates = [m for m in MODEL_REGISTRY if m != model_used]
    if not candidates:
        return decision  # only one model in the registry — nothing else to switch to

    new_model = candidates[0]
    decision["next_model"] = new_model
    decision["next_params"] = get_defaults(new_model)
    decision["reason"] = (
        f"{decision.get('reason', '')} "
        f"[Overridden: forced switch to {new_model} — {model_used} showed "
        f"< 0.01 improvement across 2 consecutive iterations.]"
    ).strip()
    decision["next_strategy"] = f"Switch to {new_model} with default params: {decision['next_params']}"
    return decision