# Prediction Evaluation Challenge

Predictive AI Evaluation Challenge — predict whether an AI subject will answer a benchmark item correctly, **without running the model on the item**.

## Problem

Given four text fields describing a (subject, item, benchmark, condition) tuple, predict the probability that the subject answers the item correctly. This is a **cold-start** prediction problem: test items have no observed responses in the training matrix.

## Repository Structure

```
├── model.py              # Required: predict() entry point for Codabench
├── labeling.py           # Optional: acquisition_function() for adaptive labeling
├── train.py              # Offline training script (produces model artifacts)
├── validate.py           # Local validation / smoke-test script
├── requirements.txt      # Python dependencies
├── models.txt            # HuggingFace model repos needed at runtime
└── artifacts/            # Trained model weights (gitignored if large)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download training data
python -c "from datasets import load_dataset; load_dataset('aims-foundations/measurement-db', split='train')"

# 3. Train offline (produces artifacts/)
python train.py

# 4. Local smoke test
python validate.py

# 5. Package for submission
zip -r submission.zip model.py labeling.py requirements.txt models.txt artifacts/
```

## Submission Format

Upload a ZIP to [Codabench](https://aimslab.stanford.edu/competition/submit) containing:

| File | Required | Purpose |
|------|----------|---------|
| `model.py` | ✅ | Must define `predict(input, labeled=None) -> float` |
| `labeling.py` | ❌ | May define `acquisition_function(input) -> float` |
| `requirements.txt` | ❌ | Python packages to install in the sandbox |
| `models.txt` | ❌ | HuggingFace model repos to pre-fetch |

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

## Rules

- Teams of 1–3 students
- One scored submission per team per calendar day (UTC)
- Sandbox is **network-isolated** at test time — no API calls from `predict()` or `acquisition_function()`
- All models must be bundled in the ZIP or declared in `models.txt`
- No state persists across rounds (fresh container each time)

## Training Data

[HuggingFace Dataset: aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)

## Grading

- **Technical report (50%)**: NeurIPS 2025 LaTeX template, 4 pages max
- **Leaderboard performance (50%)**: Best negative log-loss across all rounds
