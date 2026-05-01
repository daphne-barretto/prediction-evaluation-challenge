"""
train.py — Offline training script.

Downloads the public training dataset, fits the predictive model, and saves
artifacts to artifacts/. Run this on your own machine before submitting.

IMPORTANT: The HuggingFace repo is a collection of Parquet tables, NOT a single
`datasets` split. Response tables must be loaded explicitly and joined with
registry tables (subjects, items, benchmarks) to produce the same four-field
input shape that predict() receives at test time.

Usage:
    python train.py
"""

import os
import json
import numpy as np


REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
ARTIFACTS_DIR = "artifacts"


def load_training_data():
    """Load the public HuggingFace training dataset with proper joins.

    The dataset is a collection of Parquet tables. We load response tables
    separately from registry tables (subjects, items, benchmarks) and join
    them to produce training examples with the same shape as predict() input.
    """
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    print("Discovering response tables...")
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )
    print(f"Found {len(response_files)} response tables")

    response_features = Features(
        {
            "subject_id": Value("string"),
            "item_id": Value("string"),
            "benchmark_id": Value("string"),
            "trial": Value("int64"),
            "test_condition": Value("string"),
            "response": Value("float64"),
            "correct_answer": Value("string"),
            "trace": Value("string"),
        }
    )

    print("Loading response data...")
    responses = load_dataset(
        REPO_ID,
        data_files=response_files,
        features=response_features,
        split="train",
    )

    print("Loading registry tables...")
    items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
    benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")

    print(
        f"Loaded {len(responses)} responses, "
        f"{len(items)} items, "
        f"{len(subjects)} subjects, "
        f"{len(benchmarks)} benchmarks"
    )

    return responses, items, subjects, benchmarks


def build_training_examples(responses, items, subjects, benchmarks):
    """Join response rows with registry tables to produce training examples
    with the same shape as predict() input."""
    items_by_id = {row["item_id"]: row for row in items}
    subjects_by_id = {row["subject_id"]: row for row in subjects}
    benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}

    def to_training_example(row):
        item = items_by_id.get(row["item_id"], {})
        subject = subjects_by_id.get(row["subject_id"], {})
        benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
        display_name = subject.get("display_name") or row["subject_id"]

        return {
            "benchmark": benchmark.get("name") or row["benchmark_id"],
            "condition": row["test_condition"] or "none",
            "subject_content": f"display_name: {display_name}",
            "item_content": item.get("content"),
            "label": row["response"],
        }

    print("Building training examples...")
    examples = [to_training_example(row) for row in responses]
    print(f"Built {len(examples)} training examples")
    return examples


def compute_baselines(examples):
    """Compute simple baseline statistics from training data."""
    subject_stats = {}
    benchmark_stats = {}

    for ex in examples:
        subj = ex["subject_content"]
        bench = ex["benchmark"]
        label = ex["label"]

        if subj not in subject_stats:
            subject_stats[subj] = {"correct": 0, "total": 0}
        subject_stats[subj]["correct"] += label
        subject_stats[subj]["total"] += 1

        if bench not in benchmark_stats:
            benchmark_stats[bench] = {"correct": 0, "total": 0}
        benchmark_stats[bench]["correct"] += label
        benchmark_stats[bench]["total"] += 1

    subject_means = {
        k: v["correct"] / v["total"] for k, v in subject_stats.items()
    }
    benchmark_means = {
        k: v["correct"] / v["total"] for k, v in benchmark_stats.items()
    }

    return subject_means, benchmark_means


def train():
    """Main training pipeline."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    responses, items, subjects, benchmarks = load_training_data()
    examples = build_training_examples(responses, items, subjects, benchmarks)
    subject_means, benchmark_means = compute_baselines(examples)

    # Save baseline statistics
    with open(os.path.join(ARTIFACTS_DIR, "subject_means.json"), "w") as f:
        json.dump(subject_means, f)
    with open(os.path.join(ARTIFACTS_DIR, "benchmark_means.json"), "w") as f:
        json.dump(benchmark_means, f)

    print(f"\nSaved {len(subject_means)} subject means")
    print(f"Saved {len(benchmark_means)} benchmark means")
    print(f"Global accuracy: {sum(subject_means.values()) / len(subject_means):.3f}")

    # ------------------------------------------------------------------
    # TODO: Add your model training here. Ideas:
    #
    # 1. Neural Collaborative Filtering (NCF):
    #    - Embed subjects and items with a sentence transformer
    #    - Train an MLP head on concatenated embeddings
    #    - Save: torch.save(ncf, "artifacts/ncf_head.pt")
    #
    # 2. Item Response Theory (IRT):
    #    - Fit 2PL/3PL IRT parameters from the response matrix
    #    - Learn a content-to-parameter map for cold-start items
    #    - Save: IRT parameters + regression weights
    #
    # 3. LLM-as-Judge (no training, just declare model in models.txt)
    #
    # 4. Ensemble: combine multiple approaches
    # ------------------------------------------------------------------

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("Next: run `python validate.py` to smoke-test your submission")


if __name__ == "__main__":
    train()
