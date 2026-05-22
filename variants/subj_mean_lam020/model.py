"""subj_mean_lam020 — sub 21 with stronger smoothing (ALPHA = 0.20).

Sub 21 ("subj_mean_lookup") used ALPHA=0.1 and scored -0.61. This variant
doubles the shrinkage toward the global mean to probe the other side of
the smoothing optimum. Same artifacts as sub 21; only ALPHA differs.

The pair (lam005, lam020) brackets sub 21 on both sides. If sub 21 is
at a local minimum, both should regress slightly. If lam020 wins, the
training-time subject means are noisier than we thought and need more
pull toward 0.645.
"""
from __future__ import annotations

import json

import numpy as np


ALPHA = 0.20  # shrinkage toward global mean (sub 21 used 0.10)


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
