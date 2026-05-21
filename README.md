# Prediction Evaluation Challenge

Predictive AI Evaluation Challenge — predict whether an AI subject will answer a benchmark item correctly, **without running the model on the item**.

> 📄 The official competition handbook lives at [`docs/Predictive_Evaluation_Challenge.pdf`](docs/Predictive_Evaluation_Challenge.pdf). When this README and the handbook disagree, the handbook (plus any course-staff clarifications) wins.

## Headline result

| Submission | NLL ↑ | AUC | What it does |
|---|---:|---:|---|
| Original course baseline | -0.70 | — | reference |
| Sub 1: full mpnet rewrite (Platt + per-subject offset) | -1.01 | 0.62 | regression — diagnostic baseline |
| Sub 2: mpnet, no Platt (offset on) | -0.93 | 0.63 | -Platt |
| Sub 3: mpnet, no Platt, no offset | -0.85 | 0.63 | -offset |
| **Sub 5: sub 3 + post-hoc T-scaling (T=4.073)** | **-0.65** | **0.63** | **+0.05 vs baseline, 0.07 from top** |

T-scaling is fit on a held-out cold-start val split via `dump_cs_logits.py` (Modal job that
reconstructs the inference-time feature matrix and minimises val NLL over a single scalar).
See [`variants/sub5_t_scaled/model.py`](variants/sub5_t_scaled/model.py) for the integration.

## Problem

Given four text fields describing a (subject, item, benchmark, condition) tuple, predict the probability that the subject answers the item correctly. This is a **cold-start** prediction problem: test items have no observed responses in the training matrix.

## Repository Structure

```
├── model.py              # Required: predict() entry point for Codabench
├── labeling.py           # Optional: acquisition_function() for adaptive labeling
├── train.py              # Offline training script (produces model artifacts)
├── train_modal.py        # Modal-cloud variant of train.py (GPU-accelerated)
├── dump_cs_logits.py     # Modal job: recompute cold-start logits + fit T*
├── submit.py             # Build / list / update submissions (enforces 64-char ZIP name)
├── ledger.py             # SQLite-backed submission ledger
├── validate.py           # Local validation / smoke-test script
├── requirements.txt      # Python dependencies
├── models.txt            # HuggingFace model repos needed at runtime
├── artifacts/            # Trained model weights (large files gitignored)
├── baseline_pkg/         # Frozen copy of the original course baseline (for reproduction)
├── variants/             # Submission-specific model.py overlays
│   ├── sub5_t_scaled/    # T-scaling integration (winning submission)
│   └── sub8_t_plus_offset/  # T-scaling + per-subject offset re-enabled
├── manifests/            # JSON manifests describing each submission
└── starting_kit/         # Official starter kit from Codabench (reference)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train offline (downloads data from HuggingFace, produces artifacts/)
python train.py

# 3. Local smoke test
python validate.py

# 4. Package for submission
zip -r submission.zip model.py labeling.py requirements.txt models.txt artifacts/
```

## Loading Training Data

The HuggingFace repo is a collection of Parquet tables, **not** a single `datasets` split. Do **not** use `load_dataset("aims-foundations/measurement-db")` directly — it may mix response tables with trace tables that have different schemas.

Load response tables explicitly and join with registry tables:

```python
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    name for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)

# See train.py for the full loading pipeline with proper feature schema.
```

See `starting_kit/README.md` for complete data loading documentation.

## Submission Format

Upload a ZIP to [Codabench](https://aimslab.stanford.edu/competition/submit) containing:

| File | Required | Purpose |
|------|----------|---------|
| `model.py` | ✅ | Must define `predict(input, labeled=None) -> float` |
| `labeling.py` | ❌ | May define `acquisition_function(input) -> float` |
| `requirements.txt` | ❌ | Python packages to install in the sandbox |
| `models.txt` | ❌ | HuggingFace model repos to pre-fetch (max 5) |

## Input Format

Each input is a dict with four string keys:

| Key | Description |
|-----|-------------|
| `"benchmark"` | Benchmark name (e.g., `"MMLU"`, `"GSM8K"`) |
| `"condition"` | Test condition (e.g., `"zero-shot"`); `"none"` if N/A |
| `"subject_content"` | Description of the AI model being evaluated |
| `"item_content"` | The question/task text |

## Metrics

- **Primary**: Negative log-loss (higher is better)
- **Secondary**: AUC-ROC (higher is better)
- Scored on N=1,000 items sampled per round with two-level stratification across data categories

## Adaptive Labeling

Each round reveals K=5 ground-truth labels per category (m=5 categories per round → 25 labels total). Use `labeling.py` to define an acquisition function that selects which items get labeled. The labeled inputs are passed to `predict()` via the `labeled` argument.

## GPU Tiers

| Max params | GPU tier | Timeout |
|------------|----------|---------|
| ≤ 1B | T4 | 30 min |
| ≤ 8B | L4 | 30 min |
| ≤ 20B | A100 | 30 min |
| ≤ 70B | A100-4 / H100 | 60 min |
| ≤ 140B | A100-8 | 60 min |
| ≤ 250B | A100-mega | 60 min |

## Rules

- Teams of 1–3 students
- One scored submission per team per calendar day (UTC)
- Sandbox is **network-isolated** at test time — no API calls
- All models must be bundled in the ZIP or declared in `models.txt`
- No state persists across rounds (fresh container each time)

## Training Data

[HuggingFace Dataset: aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)

## Grading

- **Technical report (50%)**: NeurIPS 2025 LaTeX template, 4 pages max
- **Leaderboard performance (50%)**: Best negative log-loss across all rounds
