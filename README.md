# Prediction Evaluation Challenge

Predictive AI Evaluation Challenge — predict whether an AI subject will answer a benchmark item correctly, **without running the model on the item**.

> 📄 The official competition handbook lives at [`docs/Predictive_Evaluation_Challenge.pdf`](docs/Predictive_Evaluation_Challenge.pdf). When this README and the handbook disagree, the handbook (plus any course-staff clarifications) wins.

## Headline result

| Submission | NLL ↑ | AUC | What it does |
|---|---:|---:|---|
| Earlier submission, commit `40976e8` (MiniLM-L3 + IRT) | -0.70 | 0.60 | earlier `40976e8` run |
| Sub 1: full mpnet rewrite (Platt + per-subject offset) | -1.01 | 0.62 | regression diagnostic |
| Sub 5: sub 3 + post-hoc T-scaling (T=4.073) | -0.65 | 0.63 | calibration recovery |
| Sub 13: 0.5·sub 5 + 0.5·subject mean | -0.61 | 0.67 | first crack of the -0.65 plateau |
| Sub 17: 0.3·sub 5 + 0.7·subject mean | -0.61 | 0.69 | (then-)AUC champion |
| Sub 21: smoothed subject-mean lookup (no MLP) | -0.61 | 0.68 | lookup ties full pipeline |
| Sub 28: 0.2·sub 5 + 0.8·subject mean | -0.60 | 0.69 | first crack of the -0.61 plateau |
| Sub 32: Ridge regression of item text → logit µ_item | -0.5977 | 0.7019 | item-text signal w/o kNN |
| **Sub 33: item-text k-NN within subject (K=20, BETA=0.4)** | **-0.5940** | **0.7083** | **new winner — first crack of -0.60 NLL** |

The new winner adds **item-level signal**: for each test (s, item) we
encode the item text with MPNet, retrieve the top-K most cosine-similar
items the same subject already answered (PCA-256 compressed
embeddings, softmax-weighted, temperature 0.05), and blend the
per-neighborhood mean label with the smoothed subject mean
(`BETA=0.4`). The 0.01-NLL gap between sub 33 (kNN + subj-mean
fallback) and sub 28 (sub-5 blend) means **item-text k-NN dominates
the MLP+Platt+T pipeline** once per-subject identity is available.
Sub 32 (Ridge on item text, no neighborhood lookup) at -0.598
confirms the signal is in the text itself, not just the
neighborhood.

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
├── baseline_pkg/         # Frozen reproduction of the 40976e8 submission (MiniLM-L3 + IRT)
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

## Reproducing any submission

Every leaderboard submission is built from a **manifest** (in `manifests/`)
plus a per-submission `model.py` overlay (in `variants/<name>/`). The same
manifest produces a byte-identical ZIP on any machine: ZIP entries are
sorted and stamped with a fixed mtime, so `sha256(zip)` is stable. The
manifest sha + git commit are recorded in `runs/ledger.db` at build time.

### Browse what we submitted

```bash
# List every submission row (with leaderboard NLL/AUC once back-filled)
python submit.py list

# Show the manifest, files, hyperparameters and leaderboard score for one
python submit.py show 33    # sub 33 = item_knn_subject (current winner)

# Or query the SQLite ledger directly
sqlite3 runs/ledger.db \
  "SELECT id, model_name, leaderboard_nll, leaderboard_auc
     FROM submissions ORDER BY leaderboard_nll ASC LIMIT 10;"
```

### Recreate a specific ZIP

```bash
# Build, e.g., sub 33 (item-text k-NN within subject):
./build_variant.sh item_knn_subject

# What this does:
#   1. Reads manifests/item_knn_subject.json
#   2. Temporarily swaps variants/item_knn_subject/model.py into the
#      repo root (so submit.py sees it as model.py).
#   3. Runs `python submit.py build manifests/item_knn_subject.json`,
#      which materializes runs/zips/<ts>__item_knn_subject__<sha>.zip
#      and inserts a new row in runs/ledger.db.
#   4. Restores the original root-level model.py.
```

After the round resolves on Codabench, back-fill the leaderboard score:

```bash
python submit.py update 33 \
  --leaderboard-nll -0.5940 \
  --leaderboard-auc 0.7083 \
  --round-id 746473
```

### Add a new variant

1. Drop a `model.py` (and optionally `labeling.py`) into
   `variants/<your_name>/`. It can either import shared artifacts from
   `artifacts/` directly or contain the prediction logic inline.
2. Add `manifests/<your_name>.json` declaring the model name,
   hyperparameters, notes, and the list of files the ZIP must contain
   (relative to repo root). Start from
   [`manifests/item_knn_subject.json`](manifests/item_knn_subject.json)
   or [`manifests/tscale_mpnet.json`](manifests/tscale_mpnet.json) as
   templates.
3. Run `./build_variant.sh <your_name>` and confirm the new row in
   `python submit.py list`. The ZIP file path appears under
   `runs/zips/` and the manifest sha is recorded in the ledger.

> **Codabench filename limit (≤ 64 chars):** `submit.py` enforces this
> before building. ZIP names use the format
> `<UTC-ts:20>__<model_name>__<short_sha[+dirty]>.zip`, so keep model
> names ≤ ~22 chars (≤ ~17 when the working tree is dirty).

### Where artifacts come from

Most submissions reuse the same `artifacts/` directory (subject means,
item embeddings, PCA components, per-subject response tables, etc.).
These are produced by:

| Script | Produces |
|---|---|
| `python train.py` | subject/global means, item-mean tables (the F1+F4 lookups powering subs 21–27) |
| `python train_modal.py` | full MLP + Platt + T pipeline on Modal GPU (sub 1 and its diagnostic descendants 2–5) |
| `python dump_cs_logits.py` | cold-start logits + fitted T\* used by sub 5 |
| `MODAL_PROFILE=cs336-2026 modal run --detach precompute_modal.py` | MPNet item embeddings (full + PCA-256), per-subject response tables, item-mean / subject×benchmark-mean lookups, ridge `item_diff_regressor.json` (subs 31–38 and the entire sub 33 family) |

Each variant's `manifest.files[]` lists exactly which artifacts go into
its ZIP, so a single train cycle supports many submissions.

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
