"""
config.py — centralised config & secret loading.

All environment variables are loaded HERE only.
No other file should call os.environ or dotenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads backend/.env

def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy backend/.env.example to backend/.env and fill it in."
        )
    return val

# ── Secrets (loaded once at import time, never re-exported as plain strings
#    into global scope beyond this module) ──────────────────────────────────
GROQ_API_KEY: str = _require("GROQ_API_KEY")

# ── Agent settings ─────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
AGENT_TEMPERATURE = 0.2       # low temp → more deterministic code gen

# Each of the 3 registered models now runs as its own parallel branch (see
# loop.py), tuning only itself — no more cross-model switching. 3 = 1 initial
# run + 2 tuning attempts per model. Chosen to keep a full 3-branch run's
# token usage comfortably within Groq's free-tier daily budget (see
# groq_client.py) across repeated test runs during development.
ITERATIONS_PER_BRANCH = 3
MAX_EVALUATOR_RETRIES = 2     # retries with feedback before falling back to a deterministic pick

# Metrics where a fixed "good enough, stop" threshold is meaningful. Absent
# for error metrics like rmse/mae, whose scale depends on the target — no
# universal threshold makes sense there.
METRIC_STOP_THRESHOLDS = {"accuracy": 0.97, "f1": 0.97}

# ── Groq rate limiting (shared across the 3 parallel branches) ────────────
# Free tier is 30 RPM / 12K TPM for llama-3.3-70b-versatile — these stay a
# bit under that so token-estimation slop doesn't tip us over the real limit.
GROQ_RPM_LIMIT = 25
GROQ_TPM_LIMIT = 10_000
GROQ_MAX_RETRIES = 3

# ── Sandbox (Docker-based, Step 2) ────────────────────────────────────────
EXEC_TIMEOUT_SECONDS = 60
MAX_METRICS_FILE_BYTES = 10_000   # cap metrics read-back to 10 KB
MAX_DATASET_BYTES = 50 * 1024 * 1024  # 50 MB upload limit

DOCKER_IMAGE = "ml-pipeline-sandbox:latest"
DOCKER_MEMORY_LIMIT = "512m"
DOCKER_CPU_LIMIT = "1"

# ── Blocked patterns for generated code validation ────────────────────────
BLOCKED_CODE_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import requests",
    "import urllib",
    "import http",
    "__import__",
    "open(",          # no arbitrary file access
    "exec(",          # no nested exec
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "vars(",
]

# ── Allowed models and their valid hyperparameter options ────────────────────────
MODEL_REGISTRY = {
    "LogisticRegression": {
        "solver_penalty_combinations": [
            {"solver": "lbfgs", "penalty": "l2"},
            {"solver": "liblinear", "penalty": "l1"},
            {"solver": "liblinear", "penalty": "l2"},
        ],
        "valid_params": ["solver", "penalty", "C", "max_iter"],
        "defaults": {
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 100
        }
    },
    "RandomForestClassifier": {
        "valid_params": ["n_estimators", "max_depth", "min_samples_split", "random_state"],
        "defaults": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        }
    },
    "DecisionTreeClassifier": {
        "valid_params": ["max_depth", "min_samples_split", "criterion", "random_state"],
        "defaults": {
            "max_depth": None,
            "min_samples_split": 2,
            "criterion": "gini",
            "random_state": 42
        }
    }
}

LOWER_IS_BETTER_METRICS = {"rmse", "mae"}
