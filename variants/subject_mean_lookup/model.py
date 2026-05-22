"""subject_mean_lookup — slide-29 rung 3: SMOOTHED subject mean only.

Look up the subject's training-time pass rate by display_name (parsed from
the first line of subject_content), then shrink toward the global mean by a
fixed factor ALPHA. Fall back to the global mean when the subject is
unseen. No item text, no benchmark adjustment, no MLP.

Slide 30 explicitly names this rung "smoothed subject mean" (not raw). The
smoothing matters: 2 of 745 known subjects have subject_mean_acc = 0.0
exactly (DeepSeek-V3.2-Exp-Thinking, gemini-2.5-flash); a raw lookup would
predict ≈0 for them and clipping to [eps, 1-eps] doesn't actually fix log
loss if the leaderboard sample contains a y=1 for that subject (NLL
contribution = log(eps), catastrophic).

    p_smoothed = (1 - ALPHA) * subject_mean + ALPHA * global_mean

ALPHA = 0.1 (10% pull toward global mean) leaves typical subjects nearly
unchanged but maps the two saturating ones to a defensible 0.064. We also
clip to [0.001, 0.999] per slide 28's explicit recommendation. The rung-3
interpretation is preserved: this is pure subject prior, just with the
"smoothed" word taken literally.
"""
from __future__ import annotations

import json

import numpy as np


ALPHA = 0.1  # shrinkage toward global mean (slide 30: "smoothed" subject mean)


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)


def _extract_display_name(subject_content: str) -> str:
    """Mirror model.py's name-extraction so lookups match the train-time keys."""
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
