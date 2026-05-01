"""
validate.py — Local smoke test for your submission.

Simulates the Codabench evaluation loop:
1. Calls acquisition_function() on candidate inputs
2. Selects top-K for labeling
3. Calls predict() with labeled data
4. Computes log-loss and AUC-ROC

Usage:
    python validate.py
"""

import math
import sys
import warnings

import numpy as np


def make_synthetic_inputs(n=50):
    """Generate synthetic test inputs for smoke testing."""
    benchmarks = ["MMLU", "GSM8K", "HumanEval", "HellaSwag", "ARC"]
    conditions = ["zero-shot", "5-shot", "chain-of-thought", "none"]
    subjects = [
        "GPT-4 (OpenAI, 1.8T params, March 2023)",
        "Claude-3 Opus (Anthropic, unknown params, March 2024)",
        "Llama-3 70B (Meta, 70B params, April 2024)",
        "Gemini Pro (Google, unknown params, Dec 2023)",
        "Mixtral 8x7B (Mistral, 47B params, Dec 2023)",
    ]
    items = [
        "What is the capital of France?",
        "Solve: 2x + 3 = 7",
        "def fibonacci(n): # complete this function",
        "The dog chased the cat because ___ was hungry.",
        "Which element has atomic number 79?",
    ]

    rng = np.random.default_rng(42)
    inputs = []
    labels = []
    for _ in range(n):
        inp = {
            "benchmark": rng.choice(benchmarks),
            "condition": rng.choice(conditions),
            "subject_content": rng.choice(subjects),
            "item_content": rng.choice(items),
        }
        inputs.append(inp)
        labels.append(int(rng.random() > 0.4))  # ~60% pass rate

    return inputs, labels


def log_loss(y_true, y_pred, eps=1e-15):
    """Compute mean negative log-loss (higher is better)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    losses = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return float(np.mean(losses))


def auc_roc(y_true, y_pred):
    """Compute AUC-ROC."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_pred))
    except (ImportError, ValueError):
        return float("nan")


def main():
    print("=" * 60)
    print("Submission Smoke Test")
    print("=" * 60)

    # Import submission modules
    try:
        from model import predict

        print("✓ model.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import model.py: {e}")
        sys.exit(1)

    try:
        from labeling import acquisition_function

        has_labeling = True
        print("✓ labeling.py imported successfully")
    except ImportError:
        has_labeling = False
        print("⊘ labeling.py not found (using random labeling)")
    except Exception as e:
        has_labeling = False
        print(f"⚠ labeling.py import error: {e} (using random labeling)")

    # Generate synthetic data
    inputs, true_labels = make_synthetic_inputs(n=50)
    K = 5  # labels per category

    print(f"\nTest set: {len(inputs)} inputs")

    # Phase 1: Acquisition (if available)
    if has_labeling:
        print("\n--- Acquisition Phase ---")
        scores = []
        for inp in inputs:
            try:
                s = acquisition_function(inp)
                assert isinstance(s, (int, float)) and math.isfinite(s), (
                    f"Non-finite score: {s}"
                )
                scores.append(s)
            except Exception as e:
                print(f"⚠ acquisition_function error: {e}")
                scores.append(0.0)

        # Select top-K for labeling
        top_k_idx = np.argsort(scores)[-K:]
        print(f"Selected {K} inputs for labeling (indices: {sorted(top_k_idx)})")
    else:
        rng = np.random.default_rng(0)
        top_k_idx = rng.choice(len(inputs), size=K, replace=False)

    # Build labeled list
    labeled = []
    for i in top_k_idx:
        d = dict(inputs[i])
        d["label"] = true_labels[i]
        labeled.append(d)

    # Phase 2: Prediction
    print("\n--- Prediction Phase ---")
    predictions = []
    errors = 0
    for i, inp in enumerate(inputs):
        try:
            p = predict(inp, labeled=labeled)
            assert isinstance(p, float), f"predict() must return float, got {type(p)}"
            assert 0.0 <= p <= 1.0, f"predict() must return value in [0,1], got {p}"
            predictions.append(p)
        except Exception as e:
            print(f"⚠ predict() error on input {i}: {e}")
            predictions.append(0.5)
            errors += 1

    # Phase 3: Scoring
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)

    nll = log_loss(y_true, y_pred)
    auc = auc_roc(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Negative log-loss:  {nll:.4f}  (higher is better)")
    print(f"  AUC-ROC:            {auc:.4f}  (higher is better)")
    print(f"  Prediction errors:  {errors}/{len(inputs)}")
    print(f"  Prediction range:   [{min(predictions):.4f}, {max(predictions):.4f}]")
    print(f"  Prediction mean:    {np.mean(predictions):.4f}")
    print(f"  True label mean:    {np.mean(true_labels):.4f}")

    if errors == 0:
        print("\n✓ All checks passed. Ready to submit!")
    else:
        print(f"\n⚠ {errors} errors encountered. Fix before submitting.")


if __name__ == "__main__":
    main()
