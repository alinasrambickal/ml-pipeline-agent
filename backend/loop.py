"""
loop.py — Main agent loop orchestration (Docker sandbox, no API/frontend yet)

Run with:
    python loop.py --dataset path/to/data.csv --task "predict whether a customer churns"
"""
 
import argparse
import json
import pandas as pd
from pathlib import Path
 
from config import MAX_ITERATIONS, MAX_EVALUATOR_RETRIES, MODEL_REGISTRY
from agents.planner import run_planner
from agents.coder import run_coder
from agents.evaluator import run_evaluator
from sandbox.executor import run_code
from registry import validate_next_params, get_defaults, best_metric_value, is_duplicate_params
from stopping import decide_continuation, is_stagnant, pick_untried_model
 
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
 
 
def run_experiment(dataset_path: str, task_description: str) -> dict:
    print("\n" + "="*60)
    print("  AUTONOMOUS ML PIPELINE AGENT")
    print("="*60)
    print(f"  Task: {task_description}")
    print(f"  Dataset: {dataset_path}")
    print("="*60 + "\n")
 
    # ── Step 1: Plan ──────────────────────────────────────────────────────
    print("[PLANNER] Analysing dataset and task...")
    dataset_desc = describe_dataset(dataset_path)
    plan = run_planner(dataset_desc, task_description)
    print(f"[PLANNER] Plan:\n{json.dumps(plan, indent=2)}\n")
 
    # ── Agent loop ────────────────────────────────────────────────────────
    all_iterations = []
    previous_metrics = None
 
    primary_metric = plan["metric"]
    # Iteration 1 always starts with the first suggested model at its defaults
    next_model = plan["suggested_models"][0]
    next_params = get_defaults(next_model)
 
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'─'*60}")
        print(f"  ITERATION {iteration} / {MAX_ITERATIONS}")
        print(f"{'─'*60}")

        # Snapshot what's actually being used this iteration — next_model/next_params
        # get overwritten below with the evaluator's decision for the *next* iteration,
        # so the record for iteration N must not read them after that reassignment.
        model_used = next_model
        params_used = next_params

        # Coder — with retry on validation failure
        print(f"[CODER] Generating training code...")
        code = None
        generation_error = None
        for attempt in range(1, 3):  # max 2 attempts per iteration
            try:
                code = run_coder(
                    plan=plan,
                    dataset_path=dataset_path,
                    iteration=iteration,
                    next_model=model_used,
                    next_params=params_used,
                    previous_results=all_iterations if all_iterations else None,
                    generation_error=generation_error,
                )
                break
            except ValueError as e:
                generation_error = str(e)
                print(f"[CODER] Attempt {attempt} failed validation: {generation_error}. Retrying...")

        if code is None:
            print(f"[CODER] All attempts failed. Skipping iteration {iteration}.")
            all_iterations.append({
                "iteration": iteration,
                "model_used": model_used,
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
 
        print(f"[CODER] Generated {len(code.splitlines())} lines of code.")
        print("\n--- Generated Code ---")
        print(code)
        print("--- End Code ---\n")
 
        # Executor
        print(f"[EXECUTOR] Running code in restricted sandbox...")
        exec_result = run_code(code, dataset_path)
 
        print(f"[EXECUTOR] stdout: {exec_result['stdout'][:500]}")
        if exec_result["stderr"]:
            print(f"[EXECUTOR] stderr: {exec_result['stderr'][:500]}")
        if exec_result["error"]:
            print(f"[EXECUTOR] ERROR: {exec_result['error'][:500]}")
        print(f"[EXECUTOR] Metrics: {exec_result['metrics']}")
 
        # Deterministic continue/stop decision — no LLM call needed for this;
        # it's fully derivable from iteration count, this run's metrics, and
        # which models have been tried (see stopping.py).
        continuation = decide_continuation(
            iteration=iteration,
            max_iterations=MAX_ITERATIONS,
            model_used=model_used,
            metrics=exec_result["metrics"],
            previous_metrics=previous_metrics,
            history=all_iterations,
            primary_metric=primary_metric,
        )

        if not continuation["should_continue"]:
            print(f"[LOOP] Stopping: {continuation['reason']}")
            all_iterations.append({
                "iteration": iteration,
                "model_used": model_used,
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

        # Continuing — the Evaluator only decides WHAT to try next, not
        # whether to. If this model just stagnated (same model as last
        # iteration, < 0.01 improvement), tell it up front so it's less
        # likely to need a retry.
        prev_model = all_iterations[-1]["model_used"] if all_iterations else None
        must_switch_from = model_used if (
            prev_model == model_used
            and is_stagnant(model_used, exec_result["metrics"], previous_metrics, primary_metric)
        ) else None

        print(f"[EVALUATOR] Reviewing results...")
        decision = None
        rejection_reason = None
        for attempt in range(1, MAX_EVALUATOR_RETRIES + 1):
            proposal = run_evaluator(
                metrics=exec_result["metrics"],
                stdout=exec_result["stdout"],
                stderr=exec_result["stderr"] + (exec_result["error"] or ""),
                iteration=iteration,
                max_iterations=MAX_ITERATIONS,
                model_used=model_used,
                previous_metrics=previous_metrics,
                history=all_iterations,
                primary_metric=primary_metric,
                must_switch_from=must_switch_from,
                rejection_reason=rejection_reason,
            )

            try:
                validate_next_params(proposal["next_model"], proposal["next_params"])
            except ValueError as e:
                rejection_reason = str(e)
                print(f"[EVALUATOR] Attempt {attempt} rejected: {rejection_reason}")
                continue

            if is_duplicate_params(proposal["next_model"], proposal["next_params"], all_iterations):
                rejection_reason = (
                    f"{proposal['next_model']} with params {proposal['next_params']} was already "
                    f"tried in a previous iteration — propose a different configuration."
                )
                print(f"[EVALUATOR] Attempt {attempt} rejected: {rejection_reason}")
                continue

            if must_switch_from and proposal["next_model"] == must_switch_from:
                rejection_reason = (
                    f"{must_switch_from} showed < 0.01 improvement across two consecutive "
                    f"iterations — you must switch to a different model class."
                )
                print(f"[EVALUATOR] Attempt {attempt} rejected: {rejection_reason}")
                continue

            decision = proposal
            break

        if decision is None:
            fallback_model = pick_untried_model(all_iterations + [{"model_used": model_used}])
            if fallback_model is None:
                fallback_model = next(m for m in MODEL_REGISTRY if m != model_used)
            decision = {
                "next_model": fallback_model,
                "next_params": get_defaults(fallback_model),
                "next_strategy": f"Deterministic fallback to {fallback_model} with defaults.",
                "reason": (
                    f"Evaluator proposals were rejected after {MAX_EVALUATOR_RETRIES} attempts; "
                    f"used deterministic fallback."
                ),
            }
            print(f"[EVALUATOR] All attempts rejected. Falling back to {fallback_model} with defaults.")

        print(f"[EVALUATOR] Decision:\n{json.dumps(decision, indent=2)}\n")

        next_model = decision["next_model"]
        next_params = decision["next_params"]

        # Record iteration — model_used/params_used are what actually produced
        # `metrics` this iteration (next_model/next_params above are the
        # decision for the iteration after this one)
        iter_record = {
            "iteration": iteration,
            "model_used": model_used,
            "params_used": params_used,
            "code": code,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "error": exec_result["error"],
            "metrics": exec_result["metrics"],
            "next_strategy": decision.get("next_strategy", ""),
            "reason": decision.get("reason", ""),
        }
        all_iterations.append(iter_record)
        previous_metrics = exec_result["metrics"] if exec_result["metrics"] else previous_metrics

    # ── Final report ──────────────────────────────────────────────────────
    best = max(
        all_iterations,
        key=lambda r: best_metric_value(r["metrics"], primary_metric),
    )
 
    report = {
        "task": task_description,
        "dataset": dataset_path,
        "plan": plan,
        "total_iterations": len(all_iterations),
        "best_iteration": best["iteration"],
        "best_metrics": best["metrics"],
        "iterations": all_iterations,
    }
 
    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60)
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
    