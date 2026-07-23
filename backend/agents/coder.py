"""
agents/coder.py

Coder agent — takes a plan + optional previous iteration context and returns
a Python training script as a string. The script is validated for dangerous
patterns before being returned to the caller.
"""

from config import GROQ_MODEL, AGENT_TEMPERATURE, BLOCKED_CODE_PATTERNS
from groq_client import call_groq
from registry import get_registry_prompt_block

SYSTEM_PROMPT = """You are an ML engineer writing sandboxed training scripts.

Given a task plan and the next model + params to use, write a complete Python training script that:
  1. Loads data from DATASET_PATH (already available as a variable — do not hardcode a path).
  2. Preprocesses: handle missing values via SimpleImputer in a Pipeline, encode categoricals, split train/test 80/20.
  3. Trains the model specified in next_model using exactly the params in next_params.
  4. Evaluates on the test set and sets a dict called `metrics` (e.g. {{"accuracy": 0.93, "f1": 0.91}}).
  5. Prints a one-line summary to stdout.

STRICT CONSTRAINTS:
  - Use ONLY: pandas, numpy, sklearn. No other imports.
  - Do NOT import os, sys, subprocess, socket, requests, urllib, http, shutil.
  - Do NOT use open(), exec(), eval(), compile(), globals(), locals().
  - Do NOT write any files. Do NOT make network calls.
  - Define X and y before feature type detection. Compute numeric_features and categorical_features from X.
  - Do NOT use df.fillna() or X.fillna(). Use SimpleImputer inside a Pipeline only.
  - `metrics` must exist as a plain dict at the end of the script.
  - Output ONLY the Python script. No markdown fences, no explanation.

MODEL CONSTRAINTS:
  - Use ONLY the model named in next_model with EXACTLY the params in next_params. Do not add, remove, or rename any params.
  - The registry below defines what is valid. Do not deviate.

{registry_block}"""


def run_coder(
    plan: dict,
    dataset_path: str,
    iteration: int,
    next_model: str,
    next_params: dict,
    previous_results: list[dict] | None = None,
    generation_error: str | None = None,
) -> str:
    rendered_prompt = SYSTEM_PROMPT.format(registry_block=get_registry_prompt_block())
    context_block = _format_previous_results(previous_results)

    error_block = ""
    if generation_error:
        error_block = f"\nYour previous attempt was rejected with this error — fix it:\n{generation_error}\n"

    user_message = (
        f"Iteration: {iteration}\n"
        f"Dataset path: {dataset_path}\n"
        f"Plan:\n{plan}\n"
        f"Instantiate the classifier exactly as: {next_model}({', '.join(f'{k}={repr(v)}' for k, v in next_params.items())})\n"
        f"Do not define next_model or next_params as variables. Write the instantiation inline in the Pipeline.\n"
        f"{context_block}"
        f"{error_block}"
    )

    response = call_groq(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=AGENT_TEMPERATURE,
        max_tokens=2048,
    )

    code = response.choices[0].message.content.strip()
    code = _strip_fences(code)
    _validate_code(code)
    return code


def _format_previous_results(results: list[dict] | None) -> str:
    if not results:
        return "Previous iterations: none (this is the first run).\n"

    lines = ["Previous iterations:"]
    for r in results:
        failed = bool(r.get("error"))
        stderr_text = r.get("stderr", "")
        # For failed runs include the tail of stderr (error is usually at the end)
        stderr_snippet = stderr_text[-1000:] if failed else stderr_text[:300]
        prefix = "EXECUTION FAILED — " if failed else ""
        lines.append(
            f"  Iteration {r['iteration']}:\n"
            f"    metrics={r.get('metrics')}\n"
            f"    stdout={r.get('stdout', '')[:300]}\n"
            f"    stderr={prefix}{stderr_snippet}\n"
            f"    evaluator_notes={r.get('next_strategy', '')}"
        )
    return "\n".join(lines) + "\n"


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _validate_code(code: str) -> None:
    """
    Lightweight static check: reject any script containing blocked patterns.
    This is a defence-in-depth layer — real isolation comes in step 2 (Docker).
    """
    lowered = code.lower()
    for pattern in BLOCKED_CODE_PATTERNS:
        if pattern.lower() in lowered:
            raise ValueError(
                f"Generated code contains blocked pattern '{pattern}'. "
                f"Refusing to execute."
            )
