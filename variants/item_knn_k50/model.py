"""item_knn_subject (Prong A) — Item-text k-NN within subject.

For each test (s, item):
1. Encode the test item with MPNet.
2. Look up subject s's training responses (item_idx → label).
3. Compute cosine similarity between test item and each training item the
   subject answered.
4. Take the top-K most similar items, weight labels by softmax over sim/T.
5. p_knn = weighted mean label.
6. Final: BETA * p_knn + (1 - BETA) * smoothed_subject_mean.

If the subject has no training responses (cold-start subject), fall back
entirely to global mean (rare; 745/745 leaderboard subjects appear in
training per sub 21's success).
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 50
TEMP = 0.05  # softmax temperature on cosine sim (smaller -> sharper top-K)
BETA = 0.4   # weight on kNN signal vs subject mean
ALPHA = 0.1  # subj-mean shrinkage toward global


_ENCODER_NAME = "all-mpnet-base-v2"
_ENCODER = None  # lazy


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)

_ITEM_EMB_F16 = np.load("artifacts/item_embeddings_pca256_f16.npy")  # (n_items, 256) f16
_PCA_COMPONENTS = np.load("artifacts/item_pca_components.npy").astype(np.float32)  # (256, 768)
_PCA_COMPONENTS_T = _PCA_COMPONENTS.T  # (768, 256) for query projection
_ITEM_EMB_F32 = _ITEM_EMB_F16.astype(np.float32)
_ITEM_NORMS = np.linalg.norm(_ITEM_EMB_F32, axis=1, keepdims=True) + 1e-8
_ITEM_EMB_NORMED = (_ITEM_EMB_F32 / _ITEM_NORMS).astype(np.float32)

_psr = np.load("artifacts/per_subject_responses.npz", allow_pickle=True)
_PSR_ITEM_IDX = _psr["item_idx"]
_PSR_LABEL = _psr["label"].astype(np.float32)
_PSR_OFFSETS = _psr["offsets"]
_PSR_INDEX = json.load(open("artifacts/per_subject_responses_index.json"))

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
    """Encode item text -> normalized PCA-256 embedding for kNN lookup."""
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T  # (256,)
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = proj / norm
    _item_emb_cache[cache_key] = emb.astype(np.float32)
    return _item_emb_cache[cache_key]


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
    emb = _encode_item(item_content)

    p_knn = _subject_knn(name, emb)
    if p_knn is None:
        p_final = p_subj
    else:
        p_final = BETA * p_knn + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
