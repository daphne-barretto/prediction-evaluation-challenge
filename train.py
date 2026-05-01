"""
train.py — Offline training script.

Downloads the public training dataset, fits the predictive model, and saves
artifacts to artifacts/. Run this on your own machine before submitting.

Usage:
    python train.py
"""

import os
import json
import numpy as np

ARTIFACTS_DIR = "artifacts"


def load_training_data():
    """Load the public HuggingFace training dataset."""
    from datasets import load_dataset

    print("Loading training data from HuggingFace...")
    ds = load_dataset("aims-foundations/measurement-db", split="train")
    print(f"Loaded {len(ds)} training examples")
    return ds


def compute_baselines(ds):
    """Compute simple baseline statistics from training data."""
    # Per-subject mean accuracy
    subject_stats = {}
    benchmark_stats = {}

    for row in ds:
        subj = row["subject_content"]
        bench = row["benchmark"]
        label = row["label"]

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

    ds = load_training_data()
    subject_means, benchmark_means = compute_baselines(ds)

    # Save baseline statistics
    with open(os.path.join(ARTIFACTS_DIR, "subject_means.json"), "w") as f:
        json.dump(subject_means, f)
    with open(os.path.join(ARTIFACTS_DIR, "benchmark_means.json"), "w") as f:
        json.dump(benchmark_means, f)

    print(f"Saved {len(subject_means)} subject means")
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
