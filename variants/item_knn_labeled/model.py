"""item_knn_subject_labeled (Prong A + E) — kNN within subject + labeled[] pool.

Extends item_knn_subject by adding the 25 labeled[] examples (revealed at
test time via the K=5-per-category acquisition) to the kNN candidate pool.

The labeled[] examples are *in-distribution*: same benchmark(s) as the
test set. Training items are *out-of-distribution* (held-out benchmarks).
So labeled[] examples should carry stronger per-item signal at equal
semantic similarity.

For each test (s, item):
1. Encode test item.
2. Candidate set = subject s's training items + all labeled[] examples.
3. Compute cosine sims.
4. Top-K with labeled[] entries given a 2x similarity boost (in logit space).
5. Softmax-weighted (T=0.05) mean of labels.
6. Final: BETA * p_knn + (1 - BETA) * smoothed_subject_mean.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1
LABELED_BOOST = 0.10  # additive boost to cosine sim for labeled[] entries


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

_item_emb_cache: dict[str, np.ndarray] = {}
_labeled_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


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
    """Encode item text -> normalized PCA-256 embedding."""
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _item_emb_cache[cache_key] = emb
    return emb


def _index_labeled(labeled: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    items = list(labeled or [])
    if not items:
        return np.zeros((0, 256), dtype=np.float32), np.zeros(0, dtype=np.float32)
    embs = np.stack([_encode_item(ex.get("item_content", "") or "")
                     for ex in items]).astype(np.float32)
    labels = np.asarray([float(ex.get("label", 0)) for ex in items], dtype=np.float32)
    return embs, labels


def _get_labeled_cache(labeled: list[dict] | None):
    if not labeled:
        return np.zeros((0, 256), dtype=np.float32), np.zeros(0, dtype=np.float32)
    key = id(labeled)
    if key not in _labeled_cache:
        _labeled_cache[key] = _index_labeled(labeled)
    return _labeled_cache[key]


def _knn_predict(name: str, query_emb: np.ndarray,
                 labeled_embs: np.ndarray,
                 labeled_labels: np.ndarray) -> float | None:
    cands_emb: list[np.ndarray] = []
    cands_lab: list[np.ndarray] = []
    cands_boost: list[np.ndarray] = []

    if name in _PSR_INDEX:
        idx = int(_PSR_INDEX[name])
        start = int(_PSR_OFFSETS[idx])
        end = int(_PSR_OFFSETS[idx + 1])
        if end - start > 0:
            item_rows = _PSR_ITEM_IDX[start:end]
            cands_emb.append(_ITEM_EMB_NORMED[item_rows])
            cands_lab.append(_PSR_LABEL[start:end])
            cands_boost.append(np.zeros(end - start, dtype=np.float32))

    if labeled_embs.shape[0] > 0:
        cands_emb.append(labeled_embs)
        cands_lab.append(labeled_labels)
        cands_boost.append(np.full(labeled_embs.shape[0], LABELED_BOOST,
                                   dtype=np.float32))

    if not cands_emb:
        return None

    emb_all = np.vstack(cands_emb)
    lab_all = np.concatenate(cands_lab)
    boost_all = np.concatenate(cands_boost)
    sims = emb_all @ query_emb + boost_all
    k = min(K, sims.shape[0])
    top_idx = np.argpartition(sims, -k)[-k:]
    top_sims = sims[top_idx]
    top_labels = lab_all[top_idx]
    weights = np.exp((top_sims - top_sims.max()) / TEMP)
    w_sum = float(weights.sum()) + 1e-8
    return float((weights * top_labels).sum() / w_sum)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)

    labeled_embs, labeled_labels = _get_labeled_cache(labeled)
    p_knn = _knn_predict(name, emb, labeled_embs, labeled_labels)
    if p_knn is None:
        p_final = p_subj
    else:
        p_final = BETA * p_knn + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
