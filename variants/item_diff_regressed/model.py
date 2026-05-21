"""item_diff_regressed (Prong B-cold-start) — Cold-start item difficulty via Ridge.

For each test (s, item):
- p_subj = smoothed subject_mean[s] (sub 21 recipe).
- p_item: if item_id in training, use empirical item_mean.
          else encode item text with MPNet and predict logit μ_item via
          pre-fit Ridge regressor (item_diff_regressor.json).
- Final: σ(logit(p_subj) + GAMMA * (predicted_logit - global_logit)).

This is the cold-start-capable version of item_diff_blend; instead of
collapsing to subject mean for unseen items, we still get a difficulty
estimate from the item text itself.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


GAMMA = 0.5
ALPHA = 0.1


_ENCODER_NAME = "all-mpnet-base-v2"
_ENCODER = None  # lazy


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)
_REG = json.load(open("artifacts/item_diff_regressor.json"))
_W = np.asarray(_REG["coef"], dtype=np.float32)
_BIAS = float(_REG["intercept"])
_GLOBAL_LOGIT = float(_REG.get("global_logit",
                               float(np.log(_GLOBAL_MEAN_ACC / (1.0 - _GLOBAL_MEAN_ACC)))))

_item_emb_cache: dict[str, np.ndarray] = {}


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer(_ENCODER_NAME)
    return _ENCODER


def _extract_display_name(subject_content: str) -> str:
    first = (subject_content or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix):
            return first[len(prefix):].strip()
    return first.strip()


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1.0 - p)))


def _encode_item(text: str) -> np.ndarray:
    if text in _item_emb_cache:
        return _item_emb_cache[text]
    emb = _get_encoder().encode([text], convert_to_numpy=True)[0].astype(np.float32)
    _item_emb_cache[text] = emb
    return emb


def _predict_item_logit(item_content: str) -> float:
    emb = _encode_item(item_content or "")
    return float(np.dot(emb, _W) + _BIAS)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    item_logit = _predict_item_logit(item_content)

    z = _logit(p_subj) + GAMMA * (item_logit - _GLOBAL_LOGIT)
    p_final = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
