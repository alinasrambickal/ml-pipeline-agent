"""
loop.py — Main agent loop orchestration (Docker sandbox, 3 parallel per-model branches)

Run with:
    python loop.py --dataset path/to/data.csv --task "predict whether a customer churns"
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pathlib import Path

from config import ITERATIONS_PER_BRANCH, MAX_EVALUATOR_RETRIES, MODEL_REGISTRY
from agents.planner import run_planner
from agents.coder import run_coder
from agents.evaluator import run_evaluator
from sandbox.executor import run_code
from registry import validate_next_params, get_defaults, best_metric_value, is_duplicate_params, perturb_defaults
from stopping import decide_continuation

import sys

# Tee output to both terminal and log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("run.log", "w")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

import atexit

def _restore_streams_and_close_log():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.flush()
    log_file.close()

atexit.register(_restore_streams_and_close_log)

# 3 branches print concurrently — one lock so a full line from one branch
# never interleaves mid-line with another branch's print.
_print_lock = threading.Lock()


def branch_log(model_name: str, message: str) -> None:
    with _print_lock:
        print(f"[{model_name}] {message}")


def describe_dataset(csv_path: str) -> str:
    """
    Build a safe, non-sensitive dataset summary to send to the Planner.

    IMPORTANT: We send only column names, dtypes, and row count — NOT the
    raw data — to avoid leaking PII or sensitive values to the LLM API.
    """
    df = pd.read_csv(csv_path, nrows=5)  # read 5 rows just for dtype inference
    full_df = pd.read_csv(csv_path)
    row_count = len(full_df)

    lines = [f"Rows: {row_count}", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = full_df[col].nunique()
        lines.append(f"  - {col} (dtype={dtype}, unique_values={n_unique})")

    return "\n".join(lines)


def run_branch(model_name: str, plan: dict, dataset_path: str, primary_metric: str) -> list[dict]:
    """
    Runs one model's independent tuning branch to completion (up to
    ITERATIONS_PER_BRANCH iterations, or an early stop on metric threshold).
    Any unexpected failure (e.g. Groq retries exhausted) is caught here so
    it only ends this branch, not the other 2 running concurrently — the
    caller always gets back whatever iterations this branch completed.
    """
    iterations: list[dict] = []
    previous_metrics = None
    next_params = get_defaults(model_name)

    try:
        for iteration in range(1, ITERATIONS_PER_BRANCH + 1):
            branch_log(model_name, f"ITERATION {iteration} / {ITERATIONS_PER_BRANCH}")
            params_used = next_params

            # Coder — with retry on validation failure
            code = None
            generation_error = None
            for attempt in range(1, 3):  # max 2 attempts per iteration
                try:
                    code = run_coder(
                        plan=plan,
                        dataset_path=dataset_path,
                        iteration=iteration,
                        next_model=model_name,
                        next_params=params_used,
                        previous_results=iterations if iterations else None,
                        generation_error=generation_error,
                    )
                    break
                except ValueError as e:
                    generation_error = str(e)
                    branch_log(model_name, f"[CODER] Attempt {attempt} failed validation: {generation_error}. Retrying...")

            if code is None:
                branch_log(model_name, "[CODER] All attempts failed. Skipping iteration.")
                iterations.append({
                    "iteration": iteration,
                    "model_used": model_name,
                    "params_used": params_used,
                    "code": "",
                    "stdout": "",
                    "stderr": "",
                    "error": generation_error,
                    "metrics": {},
                    "next_strategy": "",
                    "reason": "Code generation failed validation after retries.",
                })
                continue

            branch_log(model_name, f"[CODER] Generated {len(code.splitlines())} lines of code.")

            # Executor
            branch_log(model_name, "[EXECUTOR] Running code in Docker sandbox...")
            exec_result = run_code(code, dataset_path)
            branch_log(model_name, f"[EXECUTOR] Metrics: {exec_result['metrics']}")
            if exec_result["error"]:
                branch_log(model_name, f"[EXECUTOR] ERROR: {exec_result['error'][:300]}")

            # Deterministic continue/stop decision — no LLM call needed.
            continuation = decide_continuation(
                iteration=iteration,
                iterations_per_branch=ITERATIONS_PER_BRANCH,
                metrics=exec_result["metrics"],
                primary_metric=primary_metric,
            )

            if not continuation["should_continue"]:
                branch_log(model_name, f"Stopping: {continuation['reason']}")
                iterations.append({
                    "iteration": iteration,
                    "model_used": model_name,
                    "params_used": params_used,
                    "code": code,
                    "stdout": exec_result["stdout"],
                    "stderr": exec_result["stderr"],
                    "error": exec_result["error"],
                    "metrics": exec_result["metrics"],
                    "next_strategy": "",
                    "reason": continuation["reason"],
                })
                break

            # Continuing — ask the Evaluator for new params for this same
            # model, retrying with feedback if its proposal gets rejected.
            branch_log(model_name, "[EVALUATOR] Reviewing results...")
            decision = None
            rejection_reason = None
            for attempt in range(1, MAX_EVALUATOR_RETRIES + 1):
                proposal = run_evaluator(
                    model_name=model_name,
                    metrics=exec_result["metrics"],
                    stdout=exec_result["stdout"],
                    stderr=exec_result["stderr"] + (exec_result["error"] or ""),
                    iteration=iteration,
                    iterations_per_branch=ITERATIONS_PER_BRANCH,
                    previous_metrics=previous_metrics,
                    history=iterations,
                    primary_metric=primary_metric,
                    rejection_reason=rejection_reason,
                )

                try:
                    validate_next_params(model_name, proposal["next_params"])
                except ValueError as e:
                    rejection_reason = str(e)
                    branch_log(model_name, f"[EVALUATOR] Attempt {attempt} rejected: {rejection_reason}")
                    continue

                # Compare against iterations + this one — `iterations` doesn't
                # have this iteration's own record yet (that's appended after
                # this decision is made), but its params_used is just as much
                # a "tried" combo as anything already in the list.
                if is_duplicate_params(model_name, proposal["next_params"], iterations + [{"model_used": model_name, "params_used": params_used}]):
                    rejection_reason = (
                        f"params {proposal['next_params']} were already tried in a previous "
                        f"iteration — propose a different configuration."
                    )
                    branch_log(model_name, f"[EVALUATOR] Attempt {attempt} rejected: {rejection_reason}")
                    continue

                decision = proposal
                break

            if decision is None:
                fallback_params = perturb_defaults(model_name, iteration)
                decision = {
                    "next_params": fallback_params,
                    "next_strategy": f"Deterministic fallback: perturbed defaults after {MAX_EVALUATOR_RETRIES} rejected proposals.",
                    "reason": f"Evaluator proposals were rejected after {MAX_EVALUATOR_RETRIES} attempts; used deterministic fallback.",
                }
                branch_log(model_name, "[EVALUATOR] All attempts rejected. Falling back to perturbed defaults.")

            branch_log(model_name, f"[EVALUATOR] Decision: {json.dumps(decision)}")

            next_params = decision["next_params"]

            iterations.append({
                "iteration": iteration,
                "model_used": model_name,
                "params_used": params_used,
                "code": code,
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"],
                "error": exec_result["error"],
                "metrics": exec_result["metrics"],
                "next_strategy": decision.get("next_strategy", ""),
                "reason": decision.get("reason", ""),
            })
            previous_metrics = exec_result["metrics"] if exec_result["metrics"] else previous_metrics

    except Exception as e:  # noqa: BLE001 — one branch failing must not take down the other 2
        branch_log(model_name, f"Branch failed with an unexpected error: {e}")

    return iterations


def run_experiment(dataset_path: str, task_description: str) -> dict:
    print("\n" + "="*60)
    print("  AUTONOMOUS ML PIPELINE AGENT")
    print("="*60)
    print(f"  Task: {task_description}")
    print(f"  Dataset: {dataset_path}")
    print("="*60 + "\n")

    # ── Plan ──────────────────────────────────────────────────────────────
    print("[PLANNER] Analysing dataset and task...")
    dataset_desc = describe_dataset(dataset_path)
    plan = run_planner(dataset_desc, task_description)
    print(f"[PLANNER] Plan:\n{json.dumps(plan, indent=2)}\n")

    primary_metric = plan["metric"]
    model_names = list(MODEL_REGISTRY.keys())

    # ── Run one branch per registered model, in parallel ────────────────────
    print(f"[LOOP] Running {len(model_names)} branches in parallel: {model_names}\n")
    branches: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
        futures = {
            pool.submit(run_branch, model_name, plan, dataset_path, primary_metric): model_name
            for model_name in model_names
        }
        for future in futures:
            branches[futures[future]] = future.result()

    all_iterations = [it for branch_iters in branches.values() for it in branch_iters]

    if not all_iterations:
        report = {
            "task": task_description,
            "dataset": dataset_path,
            "plan": plan,
            "branches": {name: {"iterations": [], "best_metrics": {}} for name in model_names},
            "best_model": None,
            "best_iteration": None,
            "best_metrics": {},
            "error": "All branches failed to produce any iterations.",
        }
        print("[LOOP] All branches failed — see per-branch logs above.")
        return report

    best = max(all_iterations, key=lambda r: best_metric_value(r["metrics"], primary_metric))

    report = {
        "task": task_description,
        "dataset": dataset_path,
        "plan": plan,
        "branches": {
            model_name: {
                "iterations": iters,
                "best_metrics": max(
                    (it["metrics"] for it in iters if it["metrics"]),
                    key=lambda m: best_metric_value(m, primary_metric),
                    default={},
                ),
            }
            for model_name, iters in branches.items()
        },
        "best_model": best["model_used"],
        "best_iteration": best["iteration"],
        "best_metrics": best["metrics"],
    }

    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60)
    print(f"  Best model:     {best['model_used']}")
    print(f"  Best iteration: {best['iteration']}")
    print(f"  Best metrics:   {best['metrics']}")
    print("="*60 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous ML Pipeline Agent")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--task", required=True, help="Plain-English task description")
    args = parser.parse_args()

    report = run_experiment(args.dataset, args.task)

    # Save report to disk
    out_path = Path("experiment_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Full report saved to {out_path}")
