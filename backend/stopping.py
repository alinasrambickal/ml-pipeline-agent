"""
stopping.py — deterministic loop-control decisions.

Each of the 3 registered models now runs as its own independent branch (see
loop.py), tuning only itself — there's no more cross-model switching, so the
only questions left are ones code can answer directly from data it already
has: has this branch's metric hit the "good enough" threshold, or has it
used up its iteration budget? Nothing here is a judgment call, so nothing
here asks the LLM.
"""

from config import METRIC_STOP_THRESHOLDS


def decide_continuation(iteration: int, iterations_per_branch: int, metrics: dict, primary_metric: str) -> dict:
    """
    Deterministically decides whether a branch continues past this
    iteration. Returns {"should_continue": bool, "reason": str} — reason is
    only set when should_continue is False.
    """
    threshold = METRIC_STOP_THRESHOLDS.get(primary_metric)
    if threshold is not None and metrics.get(primary_metric, float("-inf")) >= threshold:
        return {
            "should_continue": False,
            "reason": f"{primary_metric} reached {metrics[primary_metric]:.3f}, meets stop threshold ({threshold}).",
        }

    if iteration >= iterations_per_branch:
        return {
            "should_continue": False,
            "reason": f"Reached this branch's iteration budget ({iterations_per_branch}).",
        }

    return {"should_continue": True, "reason": ""}
