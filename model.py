"""
model.py — Required entry point for the Predictive AI Evaluation Challenge.

Defines predict(), which is called once per test input per evaluation round.
Load expensive resources (models, embeddings, weights) at module scope so they
are initialized once when the container starts.
"""

import json
import numpy as np

# ---------------------------------------------------------------------------
# Module-level state (loaded once when the container imports this file)
# ---------------------------------------------------------------------------

# TODO: Replace with your trained model artifacts.
# Example: load a trained MLP head, embedding cache, IRT parameters, etc.
#
# from sentence_transformers import SentenceTransformer
# import torch
# ENCODER = SentenceTransformer("all-mpnet-base-v2")
# NCF = torch.load("artifacts/ncf_head.pt", map_location="cpu")
# NCF.eval()

# Cached per-round state for adaptive labeling calibration
_round_calibrated = False
_calibration_offset = 0.0


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict the probability that a subject passes an item.

    Parameters
    ----------
    input : dict
        Keys: "benchmark", "condition", "subject_content", "item_content"
        (all Python str).
    labeled : list[dict] | None
        List of dicts with the same four keys plus "label" (int in {0, 1}),
        revealed by the adaptive-labeling mechanism. Same list is passed on
        every call within a round. May be empty or None.

    Returns
    -------
    float
        Probability in [0, 1] that the subject answers the item correctly.
        Must be a native Python float (not numpy/torch scalar).
    """
    global _round_calibrated, _calibration_offset

    # ------------------------------------------------------------------
    # Baseline: per-subject mean accuracy from labeled data, else 0.5
    # Replace this with your actual model logic.
    # ------------------------------------------------------------------
    base_prob = 0.5

    if labeled:
        # Simple calibration: compute mean accuracy from revealed labels
        labels = [d["label"] for d in labeled]
        base_prob = sum(labels) / len(labels) if labels else 0.5

        # Slightly adjust toward benchmark-specific accuracy if available
        same_benchmark = [
            d["label"] for d in labeled if d["benchmark"] == input["benchmark"]
        ]
        if same_benchmark:
            benchmark_mean = sum(same_benchmark) / len(same_benchmark)
            # Blend global and benchmark-specific estimates
            base_prob = 0.4 * base_prob + 0.6 * benchmark_mean

    # Clamp to avoid log-loss explosion
    return float(np.clip(base_prob, 0.01, 0.99))
