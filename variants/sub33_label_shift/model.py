"""sub33_label_shift — sub 33 + global logit-space shift from labeled[].

Extends sub 33 (item-text k-NN within subject, K=20, BETA=0.4, TEMP=0.05)
with the SAFEST possible consumption of the K=25 adaptive labels:

1. For each labeled[] example, compute sub 33's base prediction p_base.
2. delta_i = logit(label_i) - logit(p_base_i)   (binary labels clipped).
3. mean_delta = mean(delta_i) over the 25 examples.
4. Ridge-shrink toward 0: shift = mean_delta * n / (n + LAMBDA).
5. Hard-cap at +/- 0.5 in logit space.
6. Apply the same shift to all test predictions in logit space.

This is sub 14's design transplanted from sub 5 to sub 33. Hypothesis:
sub 33 is already well-calibrated (its leaderboard NLL -0.594 is the
team best, the F2 bias is small), so the shift will be near-zero and
a no-op. If it materially helps, the cap stops a sub-8-style blowup
even when 25 noisy labels happen to land in a hard cluster.

No labeling.py shipped: platform falls back to random K=5 per-category
acquisition, which matches sub 33's submission protocol.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1

LAMBDA = 25.0
SHIFT_CAP = 0.5


_ENCODER_NAME = "all-mpnet-base-v2"
_ENCODER = None  # lazy


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)

_ITEM_EMB_F16 = np.load("artifacts/item_embeddings_pca256_f16.npy")
_PCA_COMPONENTS = np.load("artifacts/item_pca_components.npy").astype(np.float32)
_PCA_COMPONENTS_T = _PCA_COMPONENTS.T
_ITEM_EMB_F32 = _ITEM_EMB_F16.astype(np.float32)
_ITEM_NORMS = np.linalg.norm(_ITEM_EMB_F32, axis=1, keepdims=True) + 1e-8
_ITEM_EMB_NORMED = (_ITEM_EMB_F32 / _ITEM_NORMS).astype(np.float32)

_psr = np.load("artifacts/per_subject_responses.npz", allow_pickle=True)
_PSR_ITEM_IDX = _psr["item_idx"]
_PSR_LABEL = _psr["label"].astype(np.float32)
_PSR_OFFSETS = _psr["offsets"]
_PSR_INDEX = json.load(open("artifacts/per_subject_responses_index.json"))

_item_emb_cache: dict[str, np.ndarray] = {}

_LABELED_CACHE_KEY = None
_LABELED_CACHE_SHIFT = 0.0


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


def _encode_item(text: str) -> np.ndarray:
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _item_emb_cache[cache_key] = emb
    return emb


def _subject_knn(name: str, query_emb: np.ndarray) -> float | None:
    if name not in _PSR_INDEX:
        return None
    idx = int(_PSR_INDEX[name])
    start = int(_PSR_OFFSETS[idx])
    end = int(_PSR_OFFSETS[idx + 1])
    if end - start == 0:
        return None
    item_rows = _PSR_ITEM_IDX[start:end]
    labels = _PSR_LABEL[start:end]
    sims = _ITEM_EMB_NORMED[item_rows] @ query_emb
    k = min(K, sims.shape[0])
    top_idx = np.argpartition(sims, -k)[-k:]
    top_sims = sims[top_idx]
    top_labels = labels[top_idx]
    weights = np.exp((top_sims - top_sims.max()) / TEMP)
    w_sum = float(weights.sum()) + 1e-8
    return float((weights * top_labels).sum() / w_sum)


def _base_predict(input: dict) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    p_knn = _subject_knn(name, emb)
    if p_knn is None:
        return p_subj
    return BETA * p_knn + (1.0 - BETA) * p_subj


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-3, 1 - 1e-3))
    return float(np.log(p / (1.0 - p)))


def _compute_shift(labeled: list[dict]) -> float:
    if not labeled:
        return 0.0
    deltas = []
    for ex in labeled:
        try:
            y = float(ex.get("label", 0))
            y_clip = max(min(y, 1.0 - 1e-3), 1e-3)
            p_base = _base_predict(ex)
            deltas.append(_logit(y_clip) - _logit(p_base))
        except Exception:
            continue
    if not deltas:
        return 0.0
    mean_delta = float(np.mean(deltas))
    n = len(deltas)
    shrunk = mean_delta * n / (n + LAMBDA)
    return float(np.clip(shrunk, -SHIFT_CAP, SHIFT_CAP))


def _cached_shift(labeled: list[dict] | None) -> float:
    global _LABELED_CACHE_KEY, _LABELED_CACHE_SHIFT
    key = id(labeled) if labeled is not None else 0
    if key != _LABELED_CACHE_KEY:
        _LABELED_CACHE_KEY = key
        _LABELED_CACHE_SHIFT = _compute_shift(labeled or [])
    return _LABELED_CACHE_SHIFT


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    p_base = _base_predict(input)
    shift = _cached_shift(labeled)
    z = _logit(p_base) + shift
    p_final = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
