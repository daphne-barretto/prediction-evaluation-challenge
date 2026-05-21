"""subj_bench_lookup (Prong C) — Subject × Benchmark cross-table lookup.

For each (subject, benchmark) pair seen in training with ≥3 responses,
use the empirical mean shrunk toward the subject's overall mean. Fall back:
  (s, b) cell  →  subject mean  →  global mean.

Hypothesis: if the leaderboard test set contains ANY benchmarks that appear
in training, the per-(s, b) cell carries information beyond the subject
mean — different subjects have different strengths on different domains.
If no benchmarks overlap (fully cold-start), this collapses to subj mean.
"""
from __future__ import annotations

import json

import numpy as np


ALPHA = 0.1  # shrinkage of subject mean toward global mean (matches sub 21)


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)
_SB = json.load(open("artifacts/subject_benchmark_mean_acc.json"))
_SB_SEP = _SB.get("sep", "|||")
_SB_PAIRS = _SB["pairs"]


def _extract_display_name(subject_content: str) -> str:
    first = (subject_content or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix):
            return first[len(prefix):].strip()
    return first.strip()


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    bench = str(input.get("benchmark", "")).strip()
    key = f"{name}{_SB_SEP}{bench}"
    if key in _SB_PAIRS:
        p = float(_SB_PAIRS[key])
    elif name in _SUBJ_MEAN_ACC:
        raw = float(_SUBJ_MEAN_ACC[name])
        p = (1.0 - ALPHA) * raw + ALPHA * _GLOBAL_MEAN_ACC
    else:
        p = _GLOBAL_MEAN_ACC
    return float(np.clip(p, 1e-3, 1 - 1e-3))
