# ML Pipeline Agent

Give it a CSV and a plain-English task, and it writes, runs, and iterates on its own scikit-learn training code until it finds a decent model, no human in the loop between iterations.
Demo
```
python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```
It plans a modeling approach, writes a full training script, runs it in a restricted sandbox, reads back the metrics and errors, and decides what to try next based on what actually happened. After up to 3 iterations it reports the best model and a full trace of every decision along the way.

## Why I built this

Most "AI writes code" demos stop once the code gets generated. The part I actually wanted to build was what happens after: reading the real output, telling a stack trace apart from a warning apart from a model that's just mediocre, and deciding what to change based on that. Closing the loop is the hard part, so that's where most of the work went.

## How it works

```
User Input (CSV + task description)
        ↓
    Planner Agent       → decides task type, target column, metric, candidate models
        ↓
    Coder Agent         → writes a sklearn training script for the chosen model/params
        ↓
  Sandbox Executor      → runs the generated code in a locked-down namespace
        ↓
  Evaluator Agent       → reads metrics/stdout/stderr, decides: continue? what next?
        ↓
    loop back to Coder (up to 3 iterations)
        ↓
  Final Report          → best model, metrics per iteration, full reasoning trace
```

Each agent (Planner, Coder, Evaluator) is one call to Groq's Llama 3.3 70B, with structured JSON output and schema checks on the way back.

## Guardrails

LLMs don't reliably follow procedural rules just because they're stated in a prompt. Anything that can be checked mechanically gets pulled out of the prompt and into plain code instead of being left up to the model's discretion:

- **Model/param validity**: a `MODEL_REGISTRY` defines every allowed model and legal hyperparameter combo (like which sklearn `solver`/`penalty` pairs are actually valid). It gets injected into every prompt so the LLM knows the rules, and every decision is checked against it in code before anything runs. Invalid combos get rejected before they ever reach the sandbox.
- **Forced model switching**: the evaluator is told to switch model classes after two iterations in a row with no real improvement. That instruction alone didn't hold up in practice, so it's enforced in code: iteration history gets used to recompute the actual improvement (not the LLM's self-reported number), and a stuck model choice gets overridden with one it hasn't tried yet.

## Sandboxing

The training code is LLM-generated, so it gets treated as untrusted input:

- Static check blocks dangerous imports/builtins (`os`, `subprocess`, `socket`, `eval`, `open`, etc.) before anything runs
- Code runs via `exec()` in a namespace with a hand-picked `__builtins__` whitelist
- 60 second wall-clock timeout via a daemon thread
- Only column names, dtypes, and row counts get sent to the LLM, raw CSV data never leaves the machine
- Metrics coming back out of the sandbox are type-checked and size-capped before anything trusts them

## Project structure

```
ml-pipeline-agent/
└── backend/
    ├── config.py             # env vars + MODEL_REGISTRY
    ├── registry.py            # registry validation, defaults, model-switch override
    ├── loop.py                 # main orchestration
    ├── agents/
    │   ├── planner.py
    │   ├── coder.py
    │   └── evaluator.py
    ├── sandbox/
    │   └── executor.py        # exec()-based sandbox (Docker planned next)
    └── data/                   # gitignored, bring your own CSV
```

## Example run

Titanic dataset (891 rows), predicting survival:

| Iteration | Model | Result |
|---|---|---|
| 1 | LogisticRegression (default params) | accuracy 0.827, F1 0.783 |
| 2 | LogisticRegression (`max_iter` increased) | accuracy 0.827, F1 0.783, no improvement |
| 3 | RandomForestClassifier (forced switch) | accuracy 0.821, F1 0.765 |

Best: accuracy 0.827 / F1 0.783 on iteration 1. Iteration 3 is the guardrail above doing its job: the evaluator wanted to keep tuning `max_iter` on a model that had already plateaued, and the override sent it to an untried model instead.

## Tech stack

| Layer | Choice |
|---|---|
| Agent brain | Groq API (Llama 3.3 70B) |
| Training | scikit-learn, pandas, numpy |
| Sandbox | restricted `exec()` namespace (Docker isolation planned) |
| Backend (planned) | FastAPI + SSE streaming + SQLite |
| Frontend (planned) | Next.js + shadcn/ui |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

cd backend
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY in .env, get one at https://console.groq.com

python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```

Run output goes to `backend/run.log`, and the final report (best model, metrics, full reasoning trace per iteration) gets saved to `backend/experiment_report.json`.

`backend/data/` is gitignored, so bring your own CSV or grab the Titanic dataset from Kaggle to reproduce the run above.

## Known limitations / what's next

- Sandbox is currently `exec()`-based, not real container isolation. Docker is next.
- No API or frontend yet. Right now it's a CLI script.
- 3 iterations max per run, hardcoded for MVP scope.
- Only handles tabular classification (CSV in, sklearn model out).
