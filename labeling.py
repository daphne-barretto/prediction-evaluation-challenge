"""
labeling.py — Optional adaptive labeling strategy.

Diversity sampling via k-means: assign each candidate to its nearest
centroid and prefer clusters not yet seen this round. This spreads the
K=5 revealed labels across distinct regions of the input space.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load once ─────────────────────────────────────────────────────────────────
_ENCODER   = SentenceTransformer("all-MiniLM-L6-v2")
_CENTROIDS = np.load("artifacts/centroids.npy")   # (64, d)

# Per-round cluster visit counter (resets each container run)
_seen: Counter = Counter()


def _to_text(input: dict) -> str:
    return (
        f"Benchmark: {input['benchmark']}\n"
        f"Condition: {input['condition']}\n"
        f"Subject: {input['subject_content']}\n"
        f"Item: {input['item_content']}"
    )


def acquisition_function(input: dict) -> float:
    """
    Score a candidate by how underrepresented its nearest cluster is.
    Higher = more desired for labeling.
    """
    try:
        x = _ENCODER.encode([_to_text(input)], convert_to_numpy=True)[0]
        c = int(np.argmin(np.linalg.norm(_CENTROIDS - x, axis=1)))
        _seen[c] += 1
        return float(-_seen[c])
    except Exception:
        return 0.0