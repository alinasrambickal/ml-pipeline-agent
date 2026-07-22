"""
stopping.py — deterministic loop-control decisions.

Whether the agent loop continues to another iteration is a property of data
the loop already has (iteration count, this run's metrics, which models have
been tried) — code can check all of it directly. The Evaluator LLM is only
called once this module has already decided to continue, and even then its
proposed next_model/next_params is validated before being trusted (see the
retry loop in loop.py). Nothing about whether to stop is left to the LLM's
judgment.
"""

from config import MODEL_REGISTRY, METRIC_STOP_THRESHOLDS
from registry import best_metric_value


def is_stagnant(model_used: str, metrics: dict, previous_metrics: dict | None, primary_metric: str) -> bool:
    """
    True if metrics show < 0.01 improvement over previous_metrics on the
    primary metric. Only meaningful when previous_metrics came from a run of
    the same model — callers are responsible for that comparison.
    """
    if not previous_metrics:
        return False
    improvement = best_metric_value(metrics, primary_metric) - best_metric_value(previous_metrics, primary_metric)
    return improvement < 0.01


def pick_untried_model(history: list[dict] | None) -> str | None:
    """Returns the first model in registry order not yet present in history's
    model_used values, or None if every registered model has been tried."""
    tried = {r.get("model_used") for r in (history or [])}
    for model_name in MODEL_REGISTRY:
        if model_name not in tried:
            return model_name
    return None


def decide_continuation(
    iteration: int,
    max_iterations: int,
    model_used: str,
    metrics: dict,
    previous_metrics: dict | None,
    history: list[dict] | None,
    primary_metric: str,
) -> dict:
    """
    Deterministically decides whether the loop continues past this iteration.
    Checked in order: metric threshold, max iterations, then "every model has
    been tried and the most recent same-model rerun didn't improve."

    Returns: {"should_continue": bool, "reason": str} — reason is only set
    when should_continue is False.
    """
    threshold = METRIC_STOP_THRESHOLDS.get(primary_metric)
    if threshold is not None and metrics.get(primary_metric, float("-inf")) >= threshold:
        return {
            "should_continue": False,
            "reason": f"{primary_metric} reached {metrics[primary_metric]:.3f}, meets stop threshold ({threshold}).",
        }

    if iteration >= max_iterations:
        return {"should_continue": False, "reason": f"Reached max iterations ({max_iterations})."}

    prev_model = history[-1].get("model_used") if history else None
    same_model_stagnant = prev_model == model_used and is_stagnant(model_used, metrics, previous_metrics, primary_metric)
    all_tried = pick_untried_model((history or []) + [{"model_used": model_used}]) is None

    if all_tried and same_model_stagnant:
        return {
            "should_continue": False,
            "reason": "All registry models have been tried; no further improvement expected.",
        }

    return {"should_continue": True, "reason": ""}
