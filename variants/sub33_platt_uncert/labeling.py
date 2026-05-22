"""labeling.py — uncertainty-based acquisition for sub33_platt_uncert.

For each candidate (subject, item) pair, return a score that is higher
when sub 33's base prediction is closer to 0.5 (maximally uncertain).
The platform reveals the top K=5 scores per data category, so within
each category we get the 5 items where sub 33 is least confident.

Why this is the right pairing for a Platt model:
* The Platt fit's information gain per label is highest near p=0.5
  (where the BCE Hessian is densest). Uncertainty sampling concentrates
  the K=25-label budget there.
* If Platt with random acquisition is a no-op (the ridge prior
  dominates), but Platt with uncertainty acquisition recovers some
  NLL, then the bottleneck was acquisition quality, not the model's
  capacity to consume labels.

If sub 33's pipeline fails (artifacts missing, encoder fails, etc.)
the function falls back to returning a constant 0.0, which makes the
platform use random per-category acquisition. We never raise.
"""
from __future__ import annotations

import json

import numpy as np


_LOADED = False
_ENCODER = None
_SUBJ_MEAN_ACC: dict = {}
_GLOBAL_MEAN_ACC = 0.5
_PCA_COMPONENTS_T = None
_ITEM_EMB_NORMED = None
_PSR_ITEM_IDX = None
_PSR_LABEL = None
_PSR_OFFSETS = None
_PSR_INDEX: dict = {}

_K = 20
_TEMP = 0.05
_BETA = 0.4
_ALPHA = 0.1

_item_emb_cache: dict = {}


def _try_load() -> bool:
    global _LOADED, _SUBJ_MEAN_ACC, _GLOBAL_MEAN_ACC
    global _PCA_COMPONENTS_T, _ITEM_EMB_NORMED
    global _PSR_ITEM_IDX, _PSR_LABEL, _PSR_OFFSETS, _PSR_INDEX
    if _LOADED:
        return True
    try:
        _SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
        _GLOBAL_MEAN_ACC = float(
            json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
        )
        item_emb_f16 = np.load("artifacts/item_embeddings_pca256_f16.npy")
        pca_components = np.load("artifacts/item_pca_components.npy").astype(np.float32)
        _PCA_COMPONENTS_T = pca_components.T
        item_emb_f32 = item_emb_f16.astype(np.float32)
        norms = np.linalg.norm(item_emb_f32, axis=1, keepdims=True) + 1e-8
        _ITEM_EMB_NORMED = (item_emb_f32 / norms).astype(np.float32)
        psr = np.load("artifacts/per_subject_responses.npz", allow_pickle=True)
        _PSR_ITEM_IDX = psr["item_idx"]
        _PSR_LABEL = psr["label"].astype(np.float32)
        _PSR_OFFSETS = psr["offsets"]
        _PSR_INDEX = json.load(open("artifacts/per_subject_responses_index.json"))
        _LOADED = True
        return True
    except Exception:
        return False


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-mpnet-base-v2")
    return _ENCODER


def _extract_display_name(subject_content: str) -> str:
    first = (subject_content or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix):
            return first[len(prefix):].strip()
    return first.strip()


def _encode_item(text: str):
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _item_emb_cache[cache_key] = emb
    return emb


def _subject_knn(name: str, query_emb):
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
    k = min(_K, sims.shape[0])
    top_idx = np.argpartition(sims, -k)[-k:]
    top_sims = sims[top_idx]
    top_labels = labels[top_idx]
    weights = np.exp((top_sims - top_sims.max()) / _TEMP)
    w_sum = float(weights.sum()) + 1e-8
    return float((weights * top_labels).sum() / w_sum)


def _base_predict(input: dict) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - _ALPHA) * raw_subj + _ALPHA * _GLOBAL_MEAN_ACC
    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    p_knn = _subject_knn(name, emb)
    if p_knn is None:
        return p_subj
    return _BETA * p_knn + (1.0 - _BETA) * p_subj


def acquisition_function(input: dict) -> float:
    """Return -|p_sub33(input) - 0.5|. Higher = more uncertain = more wanted."""
    if not _try_load():
        return 0.0
    try:
        p = _base_predict(input)
        return float(-abs(p - 0.5))
    except Exception:
        return 0.0
