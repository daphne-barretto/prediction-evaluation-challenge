"""sub33_platt_uncert — same Platt model.py as sub33_label_platt,
paired with an UNCERTAINTY-based labeling.py acquisition function.

See labeling.py for the acquisition logic. The model.py is byte-identical
to variants/sub33_label_platt/model.py so that the only difference between
this submission and sub33_label_platt is which 25 labels the platform
reveals (informative-uncertainty vs random).
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


K = 20
TEMP = 0.05
BETA = 0.4
ALPHA = 0.1

LAMBDA_A = 10.0
LAMBDA_B = 10.0
NEWTON_STEPS = 5
A_CLAMP = (0.5, 2.0)


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

_LABELED_CACHE_KEY = None
_LABELED_CACHE_AB = (1.0, 0.0)


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


def _fit_platt(labeled: list[dict]) -> tuple[float, float]:
    """Newton-Raphson fit of (a, b) with ridge prior toward identity."""
    if not labeled:
        return 1.0, 0.0
    zs, ys = [], []
    for ex in labeled:
        try:
            y = float(ex.get("label", 0))
            ys.append(max(min(y, 1.0), 0.0))
            zs.append(_logit(_base_predict(ex)))
        except Exception:
            continue
    if not zs:
        return 1.0, 0.0
    z = np.asarray(zs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(NEWTON_STEPS):
        eta = a * z + b
        p = 1.0 / (1.0 + np.exp(-eta))
        r = p - y
        w = p * (1.0 - p)
        g_a = float((r * z).sum() + 2.0 * LAMBDA_A * (a - 1.0))
        g_b = float(r.sum() + 2.0 * LAMBDA_B * b)
        H_aa = float((w * z * z).sum() + 2.0 * LAMBDA_A)
        H_bb = float(w.sum() + 2.0 * LAMBDA_B)
        H_ab = float((w * z).sum())
        det = H_aa * H_bb - H_ab * H_ab
        if det <= 1e-9:
            break
        inv = np.array([[H_bb, -H_ab], [-H_ab, H_aa]]) / det
        delta = inv @ np.array([g_a, g_b])
        a -= float(delta[0])
        b -= float(delta[1])
        a = float(np.clip(a, A_CLAMP[0], A_CLAMP[1]))
    return a, b


def _cached_ab(labeled: list[dict] | None) -> tuple[float, float]:
    global _LABELED_CACHE_KEY, _LABELED_CACHE_AB
    key = id(labeled) if labeled is not None else 0
    if key != _LABELED_CACHE_KEY:
        _LABELED_CACHE_KEY = key
        _LABELED_CACHE_AB = _fit_platt(labeled or [])
    return _LABELED_CACHE_AB


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    p_base = _base_predict(input)
    a, b = _cached_ab(labeled)
    z = a * _logit(p_base) + b
    p_final = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
