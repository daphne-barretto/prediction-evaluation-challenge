"""
labeling.py — baseline_pkg/labeling.py

Pure k-means diversity acquisition (matches commit 40976e8's labeling.py).
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

_ENCODER   = SentenceTransformer("paraphrase-MiniLM-L3-v2")
_CENTROIDS = np.load("artifacts/centroids.npy")

_seen: Counter = Counter()


def _to_text(input: dict) -> str:
    return (
        f"Benchmark: {input['benchmark']}\n"
        f"Condition: {input['condition']}\n"
        f"Subject: {input['subject_content']}\n"
        f"Item: {input['item_content']}"
    )


def acquisition_function(input: dict) -> float:
    try:
        x = _ENCODER.encode([_to_text(input)], convert_to_numpy=True)[0]
        c = int(np.argmin(np.linalg.norm(_CENTROIDS - x, axis=1)))
        _seen[c] += 1
        return float(-_seen[c])
    except Exception:
        return 0.0
