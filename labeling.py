"""
labeling.py — Adaptive labeling acquisition function.

Score = uncertainty * diversity:
  - uncertainty = 1 - |2 * p - 1|, where p is the model's predicted
    probability (so values near 0.5 score highest).
  - diversity   = 1 / (1 + visits_to_nearest_cluster), where clusters come
    from the k-means centroids fit on item embeddings during training.

The encoder (all-mpnet-base-v2) and predict() are reused from model.py so
the embedding dimension and scoring are always consistent with the model.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# Reuse the already-loaded encoder + predict() from model.py. Importing model
# triggers its module-level init (encoder + MLP + artifacts), so we don't pay
# that cost twice.
from model import _ENCODER, predict

_CENTROIDS = np.load("artifacts/centroids.npy")   # (n_clusters, 768)

# Per-round cluster visit counter (resets each container run).
_seen: Counter = Counter()


def _to_text(input: dict) -> str:
    return (
        f"Benchmark: {input.get('benchmark', '')}\n"
        f"Condition: {input.get('condition', '')}\n"
        f"Subject: {input.get('subject_content', '')}\n"
        f"Item: {input.get('item_content', '')}"
    )


def acquisition_function(input: dict) -> float:
    """Higher score = more desired for labeling."""
    try:
        x = _ENCODER.encode([_to_text(input)], convert_to_numpy=True)[0]
        c = int(np.argmin(np.linalg.norm(_CENTROIDS - x, axis=1)))
        _seen[c] += 1
        diversity = 1.0 / (1.0 + _seen[c])

        p = float(predict(input, labeled=None))
        uncertainty = 1.0 - abs(2.0 * p - 1.0)

        return float(uncertainty * diversity)
    except Exception:
        return 0.0
