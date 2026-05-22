"""subj_bayes_K — sub 21 with per-subject Beta-Bernoulli posterior using K labels.

Builds on sub 21 (smoothed subject-mean lookup, NLL=-0.61). Adds a
principled Bayesian update from the K=5 labeled examples revealed at
test time, instead of using them as an additive residual shift (which
sub 8, 14, 20 all showed is too noisy).

For each test input (subject s):
  prior:        α0 = ȳ_s · n_prior,  β0 = (1 - ȳ_s) · n_prior
  observe k_pos positives out of K_s labels for that subject in `labeled`
  posterior mean: p̂ = (α0 + k_pos) / (α0 + β0 + K_s)

Smoothing toward global mean is applied to the posterior, matching sub 21.
N_PRIOR = 25 gives K=5 a weight of 5/30 ≈ 17% — large enough to move the
prediction when the K labels disagree with the training-time mean,
small enough to avoid the high-variance failures seen in sub 8/14/20.

When K_s = 0 (no labels for this subject), this collapses exactly to
sub 21's prediction. So the worst case is "tie sub 21 at -0.61".

Hypothesis: any cold-start subjects (no training data) or
disagreement-prone subjects will benefit from the K signal, gaining
0.005-0.02 NLL on the leaderboard.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np


N_PRIOR = 25.0          # prior strength (≈ effective training samples per subject)
ALPHA_SHRINK = 0.10     # shrinkage toward global (matches sub 21)


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


_labeled_by_subject: dict[int, dict[str, tuple[int, int]]] = {}


def _index_labeled(labeled: Iterable[dict]) -> dict[str, tuple[int, int]]:
    """Group K=5 labels by subject display_name. Returns {name: (k_pos, k_total)}."""
    out: dict[str, list[int]] = {}
    for ex in labeled or []:
        name = _extract_display_name(ex.get("subject_content", ""))
        try:
            y = int(ex.get("label", 0))
        except (TypeError, ValueError):
            continue
        out.setdefault(name, []).append(y)
    return {n: (sum(ys), len(ys)) for n, ys in out.items()}


def _get_subject_labels(labeled: list[dict] | None, name: str) -> tuple[int, int]:
    if not labeled:
        return (0, 0)
    key = id(labeled)
    if key not in _labeled_by_subject:
        _labeled_by_subject[key] = _index_labeled(labeled)
    return _labeled_by_subject[key].get(name, (0, 0))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    prior_mean = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    k_pos, k_total = _get_subject_labels(labeled, name)

    alpha0 = prior_mean * N_PRIOR
    beta0 = (1.0 - prior_mean) * N_PRIOR
    posterior = (alpha0 + k_pos) / (alpha0 + beta0 + k_total)

    smoothed = (1.0 - ALPHA_SHRINK) * posterior + ALPHA_SHRINK * _GLOBAL_MEAN_ACC
    return float(np.clip(smoothed, 1e-3, 1 - 1e-3))
