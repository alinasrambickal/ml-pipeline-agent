"""
registry.py

Single source of truth for MODEL_REGISTRY logic.
All agents and loop.py import from here — never directly from config.
"""

from config import MODEL_REGISTRY, LOWER_IS_BETTER_METRICS


def get_registry_prompt_block() -> str:
    """
    Returns a formatted string describing the full registry — used by the
    Coder, which needs to know the shape of next_model/next_params it's
    handed regardless of which model that turns out to be.
    """
    lines = ["ALLOWED MODELS AND PARAMETERS:"]
    for model_name, spec in MODEL_REGISTRY.items():
        lines.append(_format_model_block(model_name, spec))
    return "\n".join(lines)


def get_model_prompt_block(model_name: str) -> str:
    """
    Returns a formatted string describing just ONE model — used by the
    Evaluator, which (per branch) only ever tunes a single fixed model and
    has no reason to see the other two in its prompt.
    """
    spec = MODEL_REGISTRY[model_name]
    return f"MODEL YOU ARE TUNING:{_format_model_block(model_name, spec)}"


def _format_model_block(model_name: str, spec: dict) -> str:
    lines = [f"\n  {model_name}:"]
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


def is_duplicate_params(model_name: str, params: dict, history: list[dict] | None) -> bool:
    """True if this exact model+params combo was already used in a previous iteration."""
    if not history:
        return False
    return any(r.get("model_used") == model_name and r.get("params_used") == params for r in history)


def perturb_defaults(model_name: str, attempt: int) -> dict:
    """
    Deterministic last-resort fallback for when the Evaluator can't produce
    a valid, non-duplicate proposal after retries: start from this model's
    defaults and nudge its first numeric param by a fixed factor. `attempt`
    scales the nudge so repeated fallbacks within the same branch (rare, but
    possible) still land on distinct configs rather than repeating.
    """
    params = get_defaults(model_name)
    for key in MODEL_REGISTRY[model_name]["valid_params"]:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        factor = 1 + 0.5 * attempt
        nudged = value * factor
        params[key] = int(round(nudged)) if isinstance(value, int) else round(nudged, 4)
        break
    return params