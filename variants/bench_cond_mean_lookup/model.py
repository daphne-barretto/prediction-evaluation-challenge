"""bench_cond_mean_lookup — slide-29 rung 4: per-benchmark mean only.

Look up the benchmark's training-time pass rate by benchmark id. Fall back
to the global train mean when the benchmark is unseen (which is exactly the
cold-start case, since the leaderboard items are sampled from benchmarks
held out of training).

The slide names "benchmark + condition" together; we ship only the
benchmark side because train_modal.py never computes a per-(benchmark,
condition) mean and we are not retraining. This is the cleanest rung-4
ablation: it shows the leaderboard *value* of per-benchmark priors on a
cold-start eval (we expect it to collapse to the global mean since every
hidden benchmark falls through). The contrast with subject_mean_lookup
illustrates why the subject side carries signal and the benchmark side does
not for this competition's cold-start regime.
"""
from __future__ import annotations

import json

import numpy as np


_BM_MEAN_ACC = json.load(open("artifacts/benchmark_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    bm = input.get("benchmark") or ""
    p = float(_BM_MEAN_ACC.get(bm, _GLOBAL_MEAN_ACC))
    # Slide 28: "Avoid exact 0 and 1; clip to e.g. [0.001, 0.999]."
    return float(np.clip(p, 1e-3, 1 - 1e-3))
