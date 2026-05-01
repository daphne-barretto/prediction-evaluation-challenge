"""
labeling.py — Optional entry point for adaptive labeling.

Defines acquisition_function(), called once per candidate input.
The platform reveals ground-truth labels for the top-K scoring inputs
per data category, then passes them as `labeled` to predict().

If this file is absent or the function raises / returns non-finite values,
K inputs per category are chosen uniformly at random instead.
"""

import hashlib
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# TODO: Replace with a more sophisticated strategy.
# Options include:
#   - Diversity sampling with pre-fit k-means centroids
#   - Uncertainty sampling based on model confidence
#   - Farthest-point sampling in embedding space
#
# Example (diversity with pre-fit centroids):
# import numpy as np
# from sentence_transformers import SentenceTransformer
# ENCODER = SentenceTransformer("all-mpnet-base-v2")
# CENTROIDS = np.load("artifacts/centroids.npy")

_seen: Counter = Counter()


def _input_key(input: dict) -> str:
    """Deterministic string key for an input dict."""
    return json.dumps(input, sort_keys=True)


def acquisition_function(input: dict) -> float:
    """
    Score how desirable it is to reveal this input's ground-truth label.

    Parameters
    ----------
    input : dict
        Keys: "benchmark", "condition", "subject_content", "item_content"
        (all Python str).

    Returns
    -------
    float
        Higher = more desirable for labeling. Must be finite and non-NaN.
    """
    # ------------------------------------------------------------------
    # Baseline: benchmark diversity — prefer inputs from benchmarks
    # we've seen fewer times so far this round.
    # ------------------------------------------------------------------
    benchmark = input["benchmark"]
    _seen[benchmark] += 1
    score = -_seen[benchmark]  # first-of-benchmark scores highest

    return float(score)
