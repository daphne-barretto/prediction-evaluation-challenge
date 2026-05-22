# Prediction Evaluation Challenge

Predictive AI Evaluation Challenge — predict whether an AI subject will answer a benchmark item correctly, without running the model on the item.

> 📄 The official competition handbook lives at [`docs/Predictive_Evaluation_Challenge.pdf`](docs/Predictive_Evaluation_Challenge.pdf). When this README and the handbook disagree, the handbook (plus any course-staff clarifications) wins.

## This repository is the team's final code submission

Per the assignment instructions and course-staff clarification, each
team's final code is graded directly from this repository (there is
no separate Gradescope upload). Slide 40 of the Part-V deck specifies
the code deliverables: `model.py` plus `labeling.py` (if used) for
the best-scoring leaderboard submission, with the auxiliary files
needed to reproduce that run. The files at the repo root reproduce
our best-scoring submission:

| Field | Value |
|---|---|
| Submission | Sub 33 — `item_knn_subject` (leaderboard run 746473) |
| Leaderboard NLL ↑ | −0.5940 |
| Leaderboard AUC | 0.7083 |
| Manifest | [`manifests/item_knn_subject.json`](manifests/item_knn_subject.json) |
| Source variant | [`variants/item_knn_subject/model.py`](variants/item_knn_subject/model.py) |

```bash
pip install -r requirements.txt
python validate.py        # smoke-test the root model.py end-to-end
```

The files that ship with the winning submission, all at the repo root:

| File | Purpose |
|---|---|
| `model.py` | `predict(input, labeled=None) -> float` — item-text k-NN within subject |
| `requirements.txt` | Sandbox Python dependencies |
| `models.txt` | HuggingFace repos to pre-fetch (mpnet sentence-transformer) |
| `artifacts/subject_mean_acc.json` | Per-subject smoothed-mean accuracy lookup |
| `artifacts/global_mean_acc.json` | Global mean accuracy (cold-start fallback) |
| `artifacts/item_embeddings_pca256_f16.npy` | PCA-256 MPNet item embeddings (53 MB, fp16) |
| `artifacts/item_pca_components.npy` | PCA basis for projecting test-item embeddings |
| `artifacts/per_subject_responses.npz` | Sparse per-subject response history |
| `artifacts/per_subject_responses_index.json` | Subject-name → row-index map |

Sub 33 does not ship a `labeling.py`: the model ignores the round's adaptive
labels (the platform falls back to random label revelation), so omitting the
file matches the file list in `manifests/item_knn_subject.json` exactly.

## Headline result

| Submission | NLL ↑ | AUC | What it does |
|---|---:|---:|---|
| Reference: team's earlier mpnet + IRT + Newton-K=5 submission | -0.70 | 0.60 | within-pipeline reference; Newton-Raphson θ-update on the K=5 labels, no post-hoc calibration |
| Sub 1: full-stack MLP + Platt + per-subject offset | -1.01 | 0.62 | high-capacity MLP+Platt+offset diagnostic (isolates calibration column under shifted distribution) |
| Sub 5: MLP + identity Platt + post-hoc T-scaling (T=4.073) | -0.65 | 0.63 | MLP backbone + calibration-column-only post-hoc fix |
| Sub 13: 0.5·sub 5 + 0.5·subject mean | -0.61 | 0.67 | α=0.5 MLP × subject-mean blend |
| Sub 17: 0.3·sub 5 + 0.7·subject mean | -0.61 | 0.69 | α=0.3 MLP × subject-mean blend (AUC peak within MLP family) |
| Sub 21: smoothed subject-mean lookup (no MLP) | -0.61 | 0.68 | per-subject prior with no encoder, MLP, or calibration — largest single lever |
| Sub 28: 0.2·sub 5 + 0.8·subject mean | -0.60 | 0.69 | α=0.2 MLP × subject-mean blend |
| Sub 32: Ridge regression of item text → logit µ_item | -0.5977 | 0.7019 | item-text signal without kNN neighborhood |
| Sub 33: item-text k-NN within subject (K=20, BETA=0.4) | -0.5940 | 0.7083 | leaderboard winner — first sub-(-0.60) NLL |

Sub 33 adds item-level signal: for each test (subject, item)
pair, we encode the item text with MPNet, retrieve the top-K most
cosine-similar items the same subject has already answered
(PCA-256 compressed embeddings, softmax-weighted, temperature
0.05), and blend the per-neighborhood mean label with the smoothed
subject mean (`BETA=0.4`). Sub 33 outperforms the
α=0.2 MLP×subject-mean blend (sub 28) by 0.01 NLL with the same
per-subject prior, so item-text k-NN dominates the MLP+Platt+T
pipeline once per-subject identity is available. Sub 32 (Ridge
on item text, no neighborhood lookup) at -0.598 confirms the
signal is in the text itself, not just the neighborhood.

## Problem

Given four text fields describing a (subject, item, benchmark, condition) tuple, predict the probability that the subject answers the item correctly. This is a cold-start prediction problem: test items have no observed responses in the training matrix.

## Repository Structure

```
├── model.py              # WINNER — root predict() entry point (sub 33: item_knn_subject)
├── requirements.txt      # Python dependencies (sandbox install)
├── models.txt            # HuggingFace model repos pre-fetched at test time
├── artifacts/            # Trained model weights / lookups (see table above)
├── variants/             # 44 per-submission model.py overlays (one dir per ledger row)
│   ├── item_knn_subject/   # sub 33 — source of truth for root model.py
│   ├── item_diff_regressed/# sub 32 — Ridge on item text
│   ├── sub5_t_scaled/      # sub 5 — T-scaled MLP baseline
│   ├── sub28_knn_combo/    # sub 28 — sub-5 + subject-mean blend
│   └── ...                 # 40+ other ablations (see `python submit.py list`)
├── manifests/            # JSON manifests, one per submission (50 total)
├── baseline_pkg/         # Frozen reproduction of the team's earlier mpnet + IRT + Newton-K=5 submission
├── train.py              # Offline training (subject/global means, item-mean lookups)
├── train_modal.py        # Modal-cloud MLP + Platt + T pipeline (subs 1–5 family)
├── precompute_modal.py   # Modal item embeddings, PCA, per-subject response tables
├── dump_cs_logits.py     # Cold-start logits + T* refit (sub 5)
├── submit.py             # Build / list / update ledger rows (enforces 64-char ZIP cap)
├── ledger.py             # SQLite-backed submission ledger
├── validate.py           # Local smoke-test of model.py + labeling.py
├── build_variant.sh      # Wrapper: swap variants/<name>/model.py → root, then `submit.py build`
├── runs/ledger.db        # Source of truth for every submission's manifest + LB score
└── starting_kit/         # Official starter kit from Codabench (reference only)
```

## Reproducing the winning submission

The root `model.py` IS sub 33. To verify it runs end-to-end on synthetic inputs:

```bash
pip install -r requirements.txt
python validate.py
```

To rebuild the exact ZIP that was uploaded to Codabench (e.g., to verify
`manifest_sha`):

```bash
./build_variant.sh item_knn_subject
# materialises runs/zips/<ts>__item_knn_subject__<sha>.zip and inserts a row
# in runs/ledger.db. The ZIP contents match the file list at the top of this
# README plus model.py, requirements.txt, models.txt.
```

## Reproducing any other submission

Every leaderboard submission is built from a manifest (in `manifests/`)
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
# Build, e.g., sub 5 (T-scaled MLP, the pre-kNN champion):
./build_variant.sh sub5_t_scaled

# What this does:
#   1. Reads manifests/sub5_t_scaled.json
#   2. Temporarily swaps variants/sub5_t_scaled/model.py into the
#      repo root (saving the root sub-33 model.py to model.py.user).
#   3. Runs `python submit.py build manifests/sub5_t_scaled.json`,
#      which materializes runs/zips/<ts>__sub5_t_scaled__<sha>.zip
#      and inserts a new row in runs/ledger.db.
#   4. Restores the root-level model.py (sub 33) via `mv -f`.
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

> Filename limit (≤ 64 chars): `submit.py` enforces this
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
its ZIP, so a single train cycle supports many submissions. Only the
6 artifacts called out at the top of this README are required for the
winning sub-33 submission; the rest are kept here to reproduce the
ablations referenced in the technical report.

## Loading Training Data

The HuggingFace repo is a collection of Parquet tables, not a single `datasets` split. Do not use `load_dataset("aims-foundations/measurement-db")` directly — it may mix response tables with trace tables that have different schemas.

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

## Submission Format (Codabench)

For each Codabench round we upload a ZIP containing:

| File | Required | Purpose |
|------|----------|---------|
| `model.py` | ✅ | Must define `predict(input, labeled=None) -> float` |
| `labeling.py` | ❌ | May define `acquisition_function(input) -> float` |
| `requirements.txt` | ❌ | Python packages to install in the sandbox |
| `models.txt` | ❌ | HuggingFace model repos to pre-fetch (max 5) |

The winning sub-33 ZIP omits `labeling.py` because the model does not use
the round's revealed labels.

## Input Format

Each input is a dict with four string keys:

| Key | Description |
|-----|-------------|
| `"benchmark"` | Benchmark name (e.g., `"MMLU"`, `"GSM8K"`) |
| `"condition"` | Test condition (e.g., `"zero-shot"`); `"none"` if N/A |
| `"subject_content"` | Description of the AI model being evaluated |
| `"item_content"` | The question/task text |

## Metrics

- Primary: Negative log-loss (higher is better)
- Secondary: AUC-ROC (higher is better)
- Scored on N=1,000 items sampled per round with two-level stratification across data categories

## Adaptive Labeling

Each round reveals K=5 ground-truth labels per category (m=5 categories per round → 25 labels total). Use `labeling.py` to define an acquisition function that selects which items get labeled. The labeled inputs are passed to `predict()` via the `labeled` argument. The winning sub-33 submission ignores the `labeled` argument (none of our top-5 leaderboard variants consumed it; see the technical report's Part V discussion).

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
- Up to 50 scored submissions per team per calendar day (UTC) during the open leaderboard rounds
- Sandbox is network-isolated at test time — no API calls
- All models must be bundled in the ZIP or declared in `models.txt`
- No state persists across rounds (fresh container each time)

## Training Data

[HuggingFace Dataset: aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)

## Grading

- Technical report (50%): NeurIPS 2025 LaTeX template, 4 pages max
- Code (50%): this repository (per the assignment, no separate Gradescope code upload). The grader reads the root `model.py` (+ `labeling.py` if present), `requirements.txt`, `models.txt`, and the listed `artifacts/` files to reproduce the best leaderboard run.
- Leaderboard performance: Best negative log-loss across all rounds, recorded in `runs/ledger.db` and shown by `python submit.py list`.

