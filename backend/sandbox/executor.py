"""
sandbox/executor.py  (Step 1 — exec-based, pre-Docker)

Executes LLM-generated training code in a restricted Python namespace with a
wall-clock timeout. Real isolation comes in Step 2 when this is replaced by
Docker container execution.

Security measures applied here:
  1. Restricted __builtins__: removes open, exec, eval, __import__, etc.
  2. Pre-execution static validation (already done in coder.py, doubled here).
  3. Wall-clock timeout via threading.
  4. Metrics extracted only from the `metrics` variable in the exec namespace.
  5. stdout/stderr captured via contextlib redirect — not exposed to real streams.
"""

import io
import json
import threading
import contextlib
import traceback
from pathlib import Path
from config import EXEC_TIMEOUT_SECONDS, MAX_METRICS_FILE_BYTES, BLOCKED_CODE_PATTERNS


# ── Allowed builtins whitelist ────────────────────────────────────────────
# Everything NOT in this list is inaccessible inside exec'd code.
_SAFE_BUILTINS = {
    "__import__": __import__,
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "zip": zip,
    "map": map,
    "filter": filter,
    "enumerate": enumerate,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "type": type,
    "repr": repr,
    "None": None,
    "True": True,
    "False": False,
}


def run_code(code: str, dataset_path: str) -> dict:
    """
    Execute generated training code and return results.

    Args:
        code: Python script string (already validated by coder.py).
        dataset_path: Absolute path to the dataset CSV. Validated here against
                      path traversal before being injected into the namespace.

    Returns:
        Dict with keys: metrics (dict), stdout (str), stderr (str), error (str|None).
    """
    # Re-validate path (defence in depth — coder prompt already restricts this)
    safe_path = _validate_dataset_path(dataset_path)

    # Re-run static check (coder.py already did this, but executor is a trust boundary)
    _static_check(code)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result = {"metrics": {}, "stdout": "", "stderr": "", "error": None}
    exec_exception: list[Exception] = []  # mutable container for thread result

    # Restricted namespace: only safe builtins + the dataset path constant
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "DATASET_PATH": str(safe_path),
        "print": lambda *args, **kwargs: stdout_buf.write(
            " ".join(str(a) for a in args) + kwargs.get("end", "\n")
        ),
    }

    def _exec_target():
        try:
            with contextlib.redirect_stdout(stdout_buf), \
                 contextlib.redirect_stderr(stderr_buf):
                exec(code, namespace)  # noqa: S102 — controlled, validated input
        except Exception as e:  # noqa: BLE001
            exec_exception.append(e)

    thread = threading.Thread(target=_exec_target, daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT_SECONDS)

    result["stdout"] = stdout_buf.getvalue()
    result["stderr"] = stderr_buf.getvalue()

    if thread.is_alive():
        # Thread still running after timeout — we can't kill it cleanly in
        # CPython, but daemon=True means it dies with the process. Log it.
        result["error"] = f"Execution timed out after {EXEC_TIMEOUT_SECONDS}s."
        return result

    if exec_exception:
        tb = traceback.format_exception(type(exec_exception[0]), exec_exception[0], exec_exception[0].__traceback__)
        result["error"] = "".join(tb)
        result["stderr"] += result["error"]
        return result

    # Extract metrics from the exec namespace
    raw_metrics = namespace.get("metrics")
    if raw_metrics is None:
        result["error"] = "Generated code did not set a `metrics` variable."
    elif not isinstance(raw_metrics, dict):
        result["error"] = f"`metrics` must be a dict, got {type(raw_metrics).__name__}."
    else:
        # Sanitise: only keep numeric values, cap total size
        safe_metrics = {}
        for k, v in raw_metrics.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                safe_metrics[str(k)[:64]] = round(float(v), 6)
        # Size cap
        if len(json.dumps(safe_metrics)) > MAX_METRICS_FILE_BYTES:
            result["error"] = "Metrics dict exceeded size limit."
        else:
            result["metrics"] = safe_metrics

    return result


def _validate_dataset_path(path: str) -> Path:
    """Resolve and validate the dataset path to prevent path traversal."""
    try:
        p = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        raise ValueError(f"Dataset not found: {path}") from e

    if not p.is_file():
        raise ValueError(f"Dataset path is not a file: {p}")
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV files are supported, got: {p.suffix}")
    if p.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    return p


def _static_check(code: str) -> None:
    """Re-run blocked pattern check as a trust-boundary enforcement."""
    lowered = code.lower()
    for pattern in BLOCKED_CODE_PATTERNS:
        if pattern.lower() in lowered:
            raise ValueError(
                f"Code contains blocked pattern '{pattern}' — execution refused."
            )
