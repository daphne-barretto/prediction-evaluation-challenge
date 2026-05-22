"""item_knn_xsubj — Cross-subject item-text k-NN.

Same as sub 33 (item_knn_subject) but the retrieval pool is GLOBAL: for
each test (s, item) we find the K nearest training responses across ALL
subjects, not just s's own responses. This directly probes the
Conclusion's claim that cross-subject retrieval is a natural next axis.

Mechanism:
  - Encode test item with MPNet, project to PCA-256, normalize.
  - Compute cosine sim against all 103983 unique items (one mat-vec).
  - For every training response r in the global pool, lift the
    per-item cos sim onto r (sims_at_response[r] = item_sims[item_idx[r]]).
  - Top-K=20 responses by sim; softmax-weighted mean of labels (τ=0.05).
  - Blend with smoothed subject mean: p = β·p_knn + (1-β)·p_subj. β=0.4.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1


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


def _encode_item(text: str) -> np.ndarray:
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = proj / norm
    _item_emb_cache[cache_key] = emb.astype(np.float32)
    return _item_emb_cache[cache_key]


def _global_knn(query_emb: np.ndarray) -> float:
    item_sims = _ITEM_EMB_NORMED @ query_emb  # (n_items,)
    sims_at_response = item_sims[_PSR_ITEM_IDX]
    k = min(K, sims_at_response.shape[0])
    top_idx = np.argpartition(sims_at_response, -k)[-k:]
    top_sims = sims_at_response[top_idx]
    top_labels = _PSR_LABEL[top_idx]
    weights = np.exp((top_sims - top_sims.max()) / TEMP)
    w_sum = float(weights.sum()) + 1e-8
    return float((weights * top_labels).sum() / w_sum)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    p_knn = _global_knn(emb)
    p_final = BETA * p_knn + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
