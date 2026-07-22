"""
sandbox/executor.py  (Step 2 — Docker-based)

Runs LLM-generated training code inside an isolated Docker container instead
of the Step 1 in-process exec(). The container gets no network, a read-only
root filesystem (except /output and a tmpfs /tmp), and memory/CPU/pid caps —
isolation enforced by the OS/container runtime rather than a Python builtins
whitelist. That whitelist still exists (in sandbox/runner.py, which runs
inside the container) as defense-in-depth, but it is no longer the primary
security boundary.

Since there's no shared process/namespace across the container boundary, the
generated code's `metrics` dict can't be read directly out of a namespace like
in Step 1. Instead: the dataset is mounted read-only, the generated script is
mounted read-only, an empty output directory is mounted read-write, and
runner.py (baked into the image) executes the script and writes
/output/result.json, which the host reads back after the container exits.
"""

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from config import (
    EXEC_TIMEOUT_SECONDS,
    MAX_METRICS_FILE_BYTES,
    BLOCKED_CODE_PATTERNS,
    DOCKER_IMAGE,
    DOCKER_MEMORY_LIMIT,
    DOCKER_CPU_LIMIT,
)


def run_code(code: str, dataset_path: str) -> dict:
    """
    Execute generated training code in a Docker container and return results.

    Args:
        code: Python script string (already validated by coder.py).
        dataset_path: Absolute path to the dataset CSV. Validated here against
                      path traversal before being mounted into the container.

    Returns:
        Dict with keys: metrics (dict), stdout (str), stderr (str), error (str|None).
    """
    safe_path = _validate_dataset_path(dataset_path)
    _static_check(code)

    result = {"metrics": {}, "stdout": "", "stderr": "", "error": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script_path = tmp / "script.py"
        script_path.write_text(code)
        output_dir = tmp / "output"
        output_dir.mkdir()

        container_name = f"ml-agent-sandbox-{uuid.uuid4().hex[:12]}"

        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", DOCKER_MEMORY_LIMIT,
            "--cpus", DOCKER_CPU_LIMIT,
            "--pids-limit", "128",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "-v", f"{script_path}:/workspace/script.py:ro",
            "-v", f"{safe_path}:/data/input.csv:ro",
            "-v", f"{output_dir}:/output:rw",
            "-e", f"MAX_METRICS_BYTES={MAX_METRICS_FILE_BYTES}",
            DOCKER_IMAGE,
        ]

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            result["error"] = f"Execution timed out after {EXEC_TIMEOUT_SECONDS}s."
            return result

        result_file = output_dir / "result.json"
        if not result_file.exists():
            # Container exited before writing output — e.g. Docker itself failed
            # to start it, or runner.py crashed outside its own try/except.
            result["error"] = (
                "Container exited without producing a result.\n"
                f"docker stderr: {proc.stderr[-2000:]}"
            )
            result["stderr"] = proc.stderr[-2000:]
            return result

        try:
            raw = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            result["error"] = "Container produced malformed result.json."
            return result

        result["stdout"] = raw.get("stdout", "")
        result["stderr"] = raw.get("stderr", "")
        result["error"] = raw.get("error")

        # Host-side trust boundary: re-check shape/size even though runner.py
        # already sanitised key/value types inside the container.
        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, dict):
            if result["error"] is None:
                result["error"] = "Generated code did not produce a valid metrics dict."
        elif len(json.dumps(raw_metrics)) > MAX_METRICS_FILE_BYTES:
            result["error"] = "Metrics dict exceeded size limit."
        else:
            result["metrics"] = raw_metrics

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
