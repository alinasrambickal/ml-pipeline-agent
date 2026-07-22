"""
sandbox/runner.py — runs INSIDE the Docker container (Step 2).

This is the containerized counterpart to Step 1's exec() logic. It reads the
generated training script from a read-only mount, executes it in the same
restricted-builtins namespace used in Step 1 (kept as defense-in-depth even
though the container boundary is now the primary isolation layer), and writes
the outcome to /output/result.json so the host can read it back after the
container exits — there's no shared process/namespace across the container
boundary the way there was with in-process exec().
"""

import io
import os
import json
import contextlib
import traceback
from pathlib import Path

SCRIPT_PATH = Path("/workspace/script.py")
DATASET_PATH = "/data/input.csv"
OUTPUT_PATH = Path("/output/result.json")
MAX_METRICS_BYTES = int(os.environ.get("MAX_METRICS_BYTES", "10000"))

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


def main() -> None:
    result = {"metrics": {}, "stdout": "", "stderr": "", "error": None}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    code = SCRIPT_PATH.read_text()
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "DATASET_PATH": DATASET_PATH,
        "print": lambda *args, **kwargs: stdout_buf.write(
            " ".join(str(a) for a in args) + kwargs.get("end", "\n")
        ),
    }

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, namespace)  # noqa: S102 — controlled, validated input
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        result["error"] = "".join(tb)

    result["stdout"] = stdout_buf.getvalue()
    result["stderr"] = stderr_buf.getvalue() + (result["error"] or "")

    if result["error"] is None:
        raw_metrics = namespace.get("metrics")
        if raw_metrics is None:
            result["error"] = "Generated code did not set a `metrics` variable."
        elif not isinstance(raw_metrics, dict):
            result["error"] = f"`metrics` must be a dict, got {type(raw_metrics).__name__}."
        else:
            safe_metrics = {}
            for k, v in raw_metrics.items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    safe_metrics[str(k)[:64]] = round(float(v), 6)
            if len(json.dumps(safe_metrics)) > MAX_METRICS_BYTES:
                result["error"] = "Metrics dict exceeded size limit."
            else:
                result["metrics"] = safe_metrics

    OUTPUT_PATH.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
