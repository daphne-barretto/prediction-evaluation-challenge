"""subj_mean_lam005 — sub 21 with less smoothing (ALPHA = 0.05).

Sub 21 ("subj_mean_lookup") used ALPHA=0.1 and scored -0.61. This variant
halves the shrinkage toward the global mean to test whether the
training-time subject means were already well-estimated and were being
over-pulled. Same artifacts as sub 21; only ALPHA differs.

If lam005 scores below -0.61, we under-shrunk in sub 21 (sub-percent
gain possible). If it ties, the smoothing parameter is flat near the
optimum. If it regresses, the two saturating subjects
(subject_mean_acc = 0.0) dominate the log loss with insufficient pull.
"""
from __future__ import annotations

import json

import numpy as np


ALPHA = 0.05  # shrinkage toward global mean (sub 21 used 0.10)


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)


def _extract_display_name(subject_content: str) -> str:
    first = (subject_content or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix):
            return first[len(prefix):].strip()
    return first.strip()


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    smoothed = (1.0 - ALPHA) * raw + ALPHA * _GLOBAL_MEAN_ACC
    return float(np.clip(smoothed, 1e-3, 1 - 1e-3))
