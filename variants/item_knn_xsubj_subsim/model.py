"""item_knn_xsubj_subsim — Cross-subject k-NN with subject-similarity reweighting.

Sub 33 retrieves within the test subject only; sub 64 (item_knn_xsubj)
retrieves globally with no subject identity at all. This variant
interpolates: it retrieves globally but reweights each retrieved
response by similarity between the test subject and the responding
subject (Gaussian on subject-mean-accuracy distance).

Mechanism for test (s, item):
  - Encode item, compute item_sims to all 103983 unique items.
  - For each training response r = (s', i', y'):
      w_item = exp((item_sims[i'] - max) / τ)
      w_subj = exp(-(μ(s) - μ(s'))^2 / (2 σ²))
      where μ(s) is the smoothed subject mean accuracy.
  - Take top-K=20 responses by (item_sims[i']) -- subject reweighting is
    applied only AFTER top-K is selected (item content drives candidate
    set, subject sim drives weighting within the candidate set).
  - p_knn = Σ (w_item · w_subj · y') / Σ (w_item · w_subj)
  - Final: p = β · p_knn + (1-β) · p_subj. β=0.4.

We need a flat (response_idx → subject_mean_acc(s')) lookup. Construct
once at load by expanding _PSR_OFFSETS → subject indices and looking up
subject_mean_acc via the index json.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1
SIGMA = 0.10  # Gaussian width on |μ(s) - μ(s')|; 0.10 ≈ one std of subj_mean_acc


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

# Build response-aligned subject_mean lookup (one float per response).
_INDEX_TO_NAME = {v: k for k, v in _PSR_INDEX.items()}
_n_responses = len(_PSR_ITEM_IDX)
_RESPONSE_SUBJ_MEAN = np.empty(_n_responses, dtype=np.float32)
for sidx in range(len(_PSR_OFFSETS) - 1):
    name = _INDEX_TO_NAME.get(sidx)
    raw = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC)) if name else _GLOBAL_MEAN_ACC
    smoothed = (1.0 - ALPHA) * raw + ALPHA * _GLOBAL_MEAN_ACC
    _RESPONSE_SUBJ_MEAN[int(_PSR_OFFSETS[sidx]):int(_PSR_OFFSETS[sidx + 1])] = smoothed

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


def _global_knn_subsim(p_subj: float, query_emb: np.ndarray) -> float:
    item_sims = _ITEM_EMB_NORMED @ query_emb
    sims_at_response = item_sims[_PSR_ITEM_IDX]
    k = min(K, sims_at_response.shape[0])
    top_idx = np.argpartition(sims_at_response, -k)[-k:]
    top_sims = sims_at_response[top_idx]
    top_labels = _PSR_LABEL[top_idx]
    top_subj_mean = _RESPONSE_SUBJ_MEAN[top_idx]
    w_item = np.exp((top_sims - top_sims.max()) / TEMP)
    diff = top_subj_mean - p_subj
    w_subj = np.exp(-(diff * diff) / (2.0 * SIGMA * SIGMA))
    w = w_item * w_subj
    w_sum = float(w.sum()) + 1e-8
    return float((w * top_labels).sum() / w_sum)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    p_knn = _global_knn_subsim(p_subj, emb)
    p_final = BETA * p_knn + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
