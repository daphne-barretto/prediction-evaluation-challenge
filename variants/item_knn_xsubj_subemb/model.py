"""item_knn_xsubj_subemb — Cross-subject k-NN reweighted by subject-embedding similarity.

Round-4 follow-up to sub 65 (item_knn_xsubj_subsim). Sub 65 reweighted
each cross-subject neighbor by Gaussian on |μ(s)-μ(s')|, the
subject-mean-accuracy gap, and lost 0.036 NLL versus sub 33. That ruled
out subject-mean as the right relevance signal. This variant tests
whether the *learned* subject embeddings (768-d, indexed by display name
via subject_emb_index.json) carry the relevance signal that subj_mean
did not.

Mechanism for test (s, item):
  - Encode item, compute item_sims to all 103983 unique items.
  - Look up θ_emb(s) — the test subject's 768-d learned embedding.
  - For each training response r = (s', i', y'):
      w_item = exp((item_sims[i'] - max) / τ)
      w_subj = exp(GAMMA * cos(θ_emb(s), θ_emb(s')))
  - Take top-K=20 responses by item_sims (subject reweighting applied
    after candidate selection, same as sub 65).
  - p_knn = Σ (w_item · w_subj · y') / Σ (w_item · w_subj)
  - Final: p = β · p_knn + (1-β) · p_subj. β=0.4.

Two flat lookups built at load time:
  - _RESPONSE_SUBJ_EMB_IDX[r] = emb_index for response r's subject.
  - subject_embeddings is L2-normalized once.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1
GAMMA = 5.0


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

_SUBJECT_EMB = np.load("artifacts/subject_embeddings.npy").astype(np.float32)
_SUBJECT_EMB_INDEX = json.load(open("artifacts/subject_emb_index.json"))
_SUBJECT_EMB_NORM = np.linalg.norm(_SUBJECT_EMB, axis=1, keepdims=True) + 1e-8
_SUBJECT_EMB_NORMED = (_SUBJECT_EMB / _SUBJECT_EMB_NORM).astype(np.float32)

_INDEX_TO_NAME = {v: k for k, v in _PSR_INDEX.items()}
_n_responses = len(_PSR_ITEM_IDX)
_RESPONSE_SUBJ_EMB_IDX = np.full(_n_responses, -1, dtype=np.int32)
for sidx in range(len(_PSR_OFFSETS) - 1):
    name = _INDEX_TO_NAME.get(sidx)
    emb_idx = _SUBJECT_EMB_INDEX.get(name, -1) if name else -1
    _RESPONSE_SUBJ_EMB_IDX[int(_PSR_OFFSETS[sidx]):int(_PSR_OFFSETS[sidx + 1])] = emb_idx

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
    emb = (proj / norm).astype(np.float32)
    _item_emb_cache[cache_key] = emb
    return emb


def _global_knn_subemb(test_subj_emb_normed, query_emb: np.ndarray):
    if test_subj_emb_normed is None:
        return None
    item_sims = _ITEM_EMB_NORMED @ query_emb
    sims_at_response = item_sims[_PSR_ITEM_IDX]
    k = min(K, sims_at_response.shape[0])
    top_idx = np.argpartition(sims_at_response, -k)[-k:]
    top_sims = sims_at_response[top_idx]
    top_labels = _PSR_LABEL[top_idx]
    top_subj_emb_idx = _RESPONSE_SUBJ_EMB_IDX[top_idx]
    valid = top_subj_emb_idx >= 0
    if not valid.any():
        return None
    top_subj_emb = _SUBJECT_EMB_NORMED[top_subj_emb_idx]
    subj_cos = top_subj_emb @ test_subj_emb_normed
    w_item = np.exp((top_sims - top_sims.max()) / TEMP)
    w_subj = np.exp(GAMMA * subj_cos)
    w = w_item * w_subj * valid.astype(np.float32)
    w_sum = float(w.sum()) + 1e-8
    return float((w * top_labels).sum() / w_sum)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    test_emb_idx = _SUBJECT_EMB_INDEX.get(name, -1)
    test_subj_emb_normed = (
        _SUBJECT_EMB_NORMED[test_emb_idx] if test_emb_idx >= 0 else None
    )

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    p_knn = _global_knn_subemb(test_subj_emb_normed, emb)
    if p_knn is None:
        p_final = p_subj
    else:
        p_final = BETA * p_knn + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
