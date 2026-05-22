"""logit_ens_33_32_21 — 3-way equal-weight logit-space ensemble.

Combines three independent signals in logit space with equal weights:

    z_final = (z_sub33 + z_sub32 + z_sub21) / 3
    p_final = sigmoid(z_final)

where:
* z_sub33 = logit(item-text kNN within subject) -- current LB winner -0.594
* z_sub32 = logit(Ridge regression of item text -> mu_item) -- LB -0.598
* z_sub21 = logit(smoothed subject_mean lookup, alpha=0.1) -- LB -0.612

Hypothesis: the three component models have different failure modes:
* sub 33 fails on cold-start subjects (no neighbors) and falls back to subj_mean.
* sub 32 has NO per-subject signal -- all variance comes from item text.
* sub 21 has NO item signal -- pure per-subject prior.

Logit-space averaging behaves like a log-likelihood pool: a component that
is extremely confident pulls the consensus only when the other two agree.
This is a beat-best stab. Prior tries:
* sub 45 (prob-space mean of 33 + 32): -0.61 (worse than sub 33)
* sub 54 (logit-space mean of 33 + 32): -0.60 (still worse than sub 33)
The new ingredient here is sub 21 as a third member providing pure
"subject prior" diversity.

No labeling.py shipped (random platform fallback).
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1
GAMMA32 = 0.5


_ENCODER_NAME = "all-mpnet-base-v2"
_ENCODER = None


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

_REG = json.load(open("artifacts/item_diff_regressor.json"))
_W32 = np.asarray(_REG["coef"], dtype=np.float32)
_BIAS32 = float(_REG["intercept"])
_GLOBAL_LOGIT = float(_REG.get("global_logit",
                               float(np.log(_GLOBAL_MEAN_ACC / (1.0 - _GLOBAL_MEAN_ACC)))))

_item_emb_full_cache: dict[str, np.ndarray] = {}
_item_emb_pca_cache: dict[str, np.ndarray] = {}


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
    p = float(np.clip(p, 1e-3, 1 - 1e-3))
    return float(np.log(p / (1.0 - p)))


def _encode_item_full(text: str) -> np.ndarray:
    """Return full 768d mpnet embedding (for sub 32's Ridge regressor)."""
    cache_key = text or ""
    if cache_key in _item_emb_full_cache:
        return _item_emb_full_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    _item_emb_full_cache[cache_key] = raw
    return raw


def _encode_item_pca(text: str, raw: np.ndarray | None = None) -> np.ndarray:
    cache_key = text or ""
    if cache_key in _item_emb_pca_cache:
        return _item_emb_pca_cache[cache_key]
    if raw is None:
        raw = _encode_item_full(cache_key)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _item_emb_pca_cache[cache_key] = emb
    return emb


def _subj_smoothed(name: str) -> float:
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    return (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC


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


def _p_sub33(name: str, emb_pca: np.ndarray, p_subj: float) -> float:
    p_knn = _subject_knn(name, emb_pca)
    if p_knn is None:
        return p_subj
    return BETA * p_knn + (1.0 - BETA) * p_subj


def _p_sub32(emb_full: np.ndarray, p_subj: float) -> float:
    item_logit = float(np.dot(emb_full, _W32) + _BIAS32)
    z = _logit(p_subj) + GAMMA32 * (item_logit - _GLOBAL_LOGIT)
    return 1.0 / (1.0 + float(np.exp(-z)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    p_subj = _subj_smoothed(name)

    item_content = input.get("item_content", "") or ""
    emb_full = _encode_item_full(item_content)
    emb_pca = _encode_item_pca(item_content, raw=emb_full)

    p33 = _p_sub33(name, emb_pca, p_subj)
    p32 = _p_sub32(emb_full, p_subj)
    p21 = p_subj

    z_avg = (_logit(p33) + _logit(p32) + _logit(p21)) / 3.0
    p_final = 1.0 / (1.0 + float(np.exp(-z_avg)))
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
