"""prong_ab_ensemble — average of item-text k-NN (sub 33) and Ridge (sub 32).

Builds the two top item-text models in one pipeline and returns their
arithmetic mean:

    p = 0.5 * p_knn  +  0.5 * p_ridge

where:
- p_knn: item_knn_subject pipeline (K=20, TEMP=0.05, BETA=0.4, ALPHA=0.1)
- p_ridge: item_diff_regressed pipeline (GAMMA=0.5, ALPHA=0.1)

Both sub-models share the same MPNet encoder and subject_mean table; the
ensemble is cheap on top of either alone.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1
GAMMA = 0.5
W_KNN = 0.5  # ensemble weight on kNN; ridge gets 1 - W_KNN


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

_REG = json.load(open("artifacts/item_diff_regressor.json"))
_W = np.asarray(_REG["coef"], dtype=np.float32)
_BIAS = float(_REG["intercept"])
_GLOBAL_LOGIT = float(
    _REG.get(
        "global_logit",
        float(np.log(_GLOBAL_MEAN_ACC / (1.0 - _GLOBAL_MEAN_ACC))),
    )
)

_raw_cache: dict[str, np.ndarray] = {}
_pca_cache: dict[str, np.ndarray] = {}


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


def _encode_raw(text: str) -> np.ndarray:
    cache_key = text or ""
    if cache_key in _raw_cache:
        return _raw_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    _raw_cache[cache_key] = raw
    return raw


def _encode_item_pca(text: str) -> np.ndarray:
    cache_key = text or ""
    if cache_key in _pca_cache:
        return _pca_cache[cache_key]
    proj = _encode_raw(cache_key) @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _pca_cache[cache_key] = emb
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


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    raw_emb = _encode_raw(item_content)
    pca_emb = _encode_item_pca(item_content)

    p_knn_raw = _subject_knn(name, pca_emb)
    if p_knn_raw is None:
        p_knn_final = p_subj
    else:
        p_knn_final = BETA * p_knn_raw + (1.0 - BETA) * p_subj

    item_logit = float(np.dot(raw_emb, _W) + _BIAS)
    z = _logit(p_subj) + GAMMA * (item_logit - _GLOBAL_LOGIT)
    p_ridge = 1.0 / (1.0 + np.exp(-z))

    p_final = W_KNN * p_knn_final + (1.0 - W_KNN) * p_ridge
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
