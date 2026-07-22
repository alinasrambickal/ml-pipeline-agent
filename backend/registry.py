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


def is_duplicate_params(model_name: str, params: dict, history: list[dict] | None) -> bool:
    """True if this exact model+params combo was already used in a previous iteration."""
    if not history:
        return False
    return any(r.get("model_used") == model_name and r.get("params_used") == params for r in history)