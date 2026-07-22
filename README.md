# ML Pipeline Agent

An autonomous agent that takes a CSV dataset and a plain-English task description, then writes, runs, evaluates, and iterates on its own machine learning training code — with no human in the loop between iterations.

You give it something like:

```
python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```

and it plans a modeling strategy, generates a full sklearn training script, executes it in a restricted sandbox, reads back the metrics and any errors, and decides what to try next — switching models or tuning hyperparameters based on what actually happened, not just what it was told to do. After up to 3 iterations, it reports the best model and a full trace of every decision it made along the way.

## Why this exists

Most "AI writes code" demos stop at code generation. This project is about the harder part: closing the loop. The agent has to read its own results, reason about failure modes (a stack trace vs. a convergence warning vs. a model that just isn't good enough), and make a real decision — then live with the consequences of that decision on the next iteration.

- **Agentic systems angle:** a genuine plan → act → observe → reflect loop, with deterministic guardrails layered on top of LLM decisions where prompting alone isn't reliable enough (see [Guardrails](#guardrails-not-just-prompting) below).
- **Systems/security angle:** untrusted, LLM-generated Python code is executed under a restricted `__builtins__` whitelist, a static blocklist, and a wall-clock timeout — sandboxing a moving, adversarial-by-construction input.

## How it works

```
User Input (CSV + task description)
        ↓
    Planner Agent       → decides task type, target column, metric, candidate models
        ↓
    Coder Agent         → writes a complete sklearn training script for the chosen model/params
        ↓
  Sandbox Executor      → runs the generated code in a locked-down namespace
        ↓
  Evaluator Agent       → reads metrics/stdout/stderr, decides: continue? what to try next?
        ↓
    loop back to Coder (up to 3 iterations)
        ↓
  Final Report          → best model, metrics per iteration, full reasoning trace
```

Each of the three agents (Planner, Coder, Evaluator) is a single call to Groq's Llama 3.3 70B, with structured JSON output and schema validation on the way back.

### Guardrails, not just prompting

LLMs are unreliable at consistently following procedural rules stated only in a prompt. Rather than trust the model to self-police, this project pushes anything mechanically checkable out of the prompt and into code:

- **Model/hyperparameter validity** — a `MODEL_REGISTRY` defines every allowed model and legal hyperparameter combination (e.g. valid sklearn `solver`/`penalty` pairs). The registry is injected into every prompt so the LLM knows the rules, and every decision is validated against it in code before execution — invalid combinations are rejected and never reach the sandbox.
- **Forced model switching** — the evaluator is told to switch model classes after two consecutive iterations with no meaningful improvement. Prompting alone didn't reliably produce this behavior, so it's enforced deterministically: iteration history is used to recompute actual improvement (not the LLM's self-reported number), and a stagnant model choice is overridden with an untried one before the next iteration runs.

### Sandboxing untrusted, LLM-generated code

Generated training scripts are treated as untrusted input:

- Static validation blocks dangerous imports/builtins (`os`, `subprocess`, `socket`, `eval`, `open`, etc.) before execution
- Code runs via `exec()` in a namespace with a hand-picked `__builtins__` whitelist
- 60-second wall-clock timeout via a daemon thread
- Only column names, dtypes, and row counts are ever sent to the LLM — raw CSV data never leaves the machine
- Metrics extracted from the sandbox are type-checked and size-capped before being trusted

## Example run

On the classic Titanic dataset (891 rows), predicting survival:

| Iteration | Model | Result |
|---|---|---|
| 1 | LogisticRegression (default params) | accuracy 0.827, F1 0.783 |
| 2 | LogisticRegression (`max_iter` increased) | accuracy 0.827, F1 0.783 — no improvement |
| 3 | RandomForestClassifier (forced switch after stagnation) | accuracy 0.821, F1 0.765 |

**Best: accuracy 0.827 / F1 0.783** (iteration 1). The forced switch on iteration 3 is the guardrail described above firing correctly — the evaluator wanted to keep tuning `max_iter` on a model that had already plateaued, and the deterministic override redirected it to an untried model class instead.

## Tech stack

| Layer | Choice |
|---|---|
| Agent brain | Groq API (Llama 3.3 70B) |
| Training | scikit-learn (pandas/numpy for data handling) |
| Sandbox | Restricted `exec()` namespace (Docker-based isolation planned) |
| Backend (planned) | FastAPI + SSE streaming + SQLite |
| Frontend (planned) | Next.js + shadcn/ui |

## Running it locally

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate

cd backend
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY in .env — get one at https://console.groq.com

python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```

Full run output is written to `backend/run.log`, and the final report (best model, metrics, and the complete reasoning trace for every iteration) is saved to `backend/experiment_report.json`.

Note: `backend/data/` is gitignored — bring your own CSV, or use scikit-learn/Kaggle's Titanic dataset to reproduce the example run above.

## Project status

This is an active, iterative build. Current state: the core agent loop (Planner → Coder → Executor → Evaluator, no Docker/API/frontend yet) is complete and verified end-to-end.

Planned next: swap the `exec()`-based sandbox for real Docker container isolation, wrap the loop in a FastAPI service with SSE streaming for live updates, and build a Next.js frontend for uploading a dataset and watching the agent work in real time.

---

Built by [Alina Srambickal](https://github.com/alinasrambickal), CS student at Carnegie Mellon (ML + Systems).
