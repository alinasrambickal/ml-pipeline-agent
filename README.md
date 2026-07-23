# ML Pipeline Agent

Give it a CSV and a plain-English task, and it writes, runs, and iterates on its own scikit-learn training code until it finds a decent model, no human in the loop between iterations.

Demo
```
python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```
It plans a modeling approach, then runs three model types (logistic regression, random forest, decision tree) in parallel, each one tuning its own hyperparameters independently based on what actually happened in its own runs. Once all three finish, it reports the best model across all of them and a full trace of every decision along the way.

## Why I built this

Most "AI writes code" demos stop once the code gets generated. The part I actually wanted to build was what happens after: reading the real output, telling a stack trace apart from a warning apart from a model that's just mediocre, and deciding what to change based on that. Closing the loop is the hard part, so that's where most of the work went.

## How it works

```
User Input (CSV + task description)
        ↓
    Planner Agent       → decides task type, target column, metric, candidate models
        ↓
  3 parallel branches, one per model type
        ↓
    Coder Agent         → writes a sklearn training script for that branch's model/params
        ↓
  Docker Sandbox        → runs the generated code in an isolated container
        ↓
  Evaluator Agent       → reads metrics/stdout/stderr, proposes new params for that model
        ↓
    loop back to Coder (up to 3 iterations per branch)
        ↓
  Final Report          → best model across all branches, metrics per iteration, full reasoning trace
```

Each agent call (Planner, Coder, Evaluator) goes to Groq's Llama 3.3 70B, with structured JSON output and schema checks on the way back. All three branches share one Groq API key, so their calls go through a rate limiter that keeps the combined traffic under the free tier's limits instead of blowing past them.

## Guardrails

LLMs don't reliably follow procedural rules just because they're stated in a prompt. Anything that can be checked mechanically gets pulled out of the prompt and into plain code instead of being left up to the model's discretion:

- **Model/param validity**: a `MODEL_REGISTRY` defines every allowed model and legal hyperparameter combo (like which sklearn `solver`/`penalty` pairs are actually valid). It gets injected into every prompt so the LLM knows the rules, and every decision is checked against it in code before anything runs. Invalid combos get rejected before they ever reach the sandbox.
- **Continue/stop decisions**: whether a branch keeps iterating used to be the LLM's call, based on a prompt rule ("stop after two flat iterations, or once the metric's good enough"). That rule wasn't reliably followed, so the decision now happens entirely in code, from the actual metrics and iteration count. The Evaluator doesn't even get called on the last iteration of a branch, since there's nothing left for it to decide.
- **No repeated params**: an Evaluator proposal that repeats a combo already tried in that branch gets rejected and retried, with the specific reason fed back so the model can correct itself, instead of quietly reusing a config that's already been tested.

## Independent tuning per model

An earlier version of this ran one lineage that could switch between model types when it stalled. Testing turned up a real problem with that: a single bad hyperparameter guess could trigger a full switch away from a model that had actually produced the best result of the run so far. Not how anyone tunes a model by hand. Now each of the three model types gets its own branch, tuned independently and in parallel, and the best result across all three wins at the end.

## Sandboxing

The training code is LLM-generated, so it gets treated as untrusted input:

- Static check blocks dangerous imports/builtins (`os`, `subprocess`, `socket`, `eval`, `open`, etc.) before anything runs
- Code executes inside a Docker container with no network access, a read-only filesystem, and memory/CPU/process caps
- 60 second timeout enforced from the host, with a hard container kill if it fires
- Only column names, dtypes, and row counts get sent to the LLM, raw CSV data never leaves the machine
- Metrics coming back out of the sandbox are type-checked and size-capped before anything trusts them

## Project structure

```
ml-pipeline-agent/
└── backend/
    ├── config.py             # env vars + MODEL_REGISTRY
    ├── registry.py            # registry validation, defaults, duplicate-params check
    ├── stopping.py             # continue/stop decision, no LLM involved
    ├── groq_client.py          # shared rate-limited Groq client
    ├── loop.py                 # main orchestration, runs 3 branches in parallel
    ├── agents/
    │   ├── planner.py
    │   ├── coder.py
    │   └── evaluator.py
    ├── sandbox/
    │   ├── executor.py        # host-side: builds/runs the Docker container
    │   ├── runner.py          # in-container: runs the generated script
    │   └── Dockerfile
    └── data/                   # gitignored, bring your own CSV
```

## Example run

Titanic dataset (891 rows), predicting survival, 3 branches x 3 iterations each:

| Model | Best iteration | Result |
|---|---|---|
| LogisticRegression | 1 (default params) | accuracy 0.832, F1 0.792 |
| RandomForestClassifier | 1 (default params) | accuracy 0.821, F1 0.765 |
| DecisionTreeClassifier | 3 (`max_depth=5`, `criterion=entropy`) | accuracy 0.827, F1 0.821 |

Best overall: LogisticRegression, accuracy 0.832 / F1 0.792. The decision tree branch is worth calling out too: its best result landed on the last iteration, after two earlier configs that didn't help. The old single-lineage design would have abandoned that model well before it got there.

## Tech stack

| Layer | Choice |
|---|---|
| Agent brain | Groq API (Llama 3.3 70B) |
| Training | scikit-learn, pandas, numpy |
| Sandbox | Docker, isolated container per run |
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

docker build -t ml-pipeline-sandbox:latest sandbox/
python loop.py --dataset data/titanic_full.csv --task "predict whether a passenger survived"
```

Run output goes to `backend/run.log`, and the final report (best model, metrics, full reasoning trace per iteration, grouped by branch) gets saved to `backend/experiment_report.json`.

`backend/data/` is gitignored, so bring your own CSV or grab the Titanic dataset from Kaggle to reproduce the run above.

## Known limitations / what's next

- No API or frontend yet. Right now it's a CLI script.
- 3 iterations per model branch, hardcoded for MVP scope, up to 9 total per run.
- Only handles tabular classification (CSV in, sklearn model out).
- The Groq rate limiter hasn't been tested under real sustained rate-limit pressure, only reasoned through.
