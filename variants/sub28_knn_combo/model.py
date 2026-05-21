"""sub28_knn_combo - combine sub 28 (alpha=0.20 blend, -0.60 NLL) with Prong A item-text kNN.

p_final = ALPHA * p_sub5 + (1 - ALPHA) * [BETA * p_knn + (1 - BETA) * smoothed_subj_mean]
       = 0.20 * p_sub5 + 0.32 * p_knn + 0.48 * subj_mean_smoothed   (when ALPHA=0.20, BETA=0.4)

Layers item-text signal into the subject-mean channel of sub 28. Falls back to
sub 28 when no kNN candidates available (cold-start subject).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


ALPHA = 0.20  # weight on sub5/MLP+Platt+T signal (from sub 28 / best fine-sweep result)
BETA  = 0.4   # weight on kNN signal within the (1-ALPHA) channel
KNN_K = 20
KNN_TEMP = 0.05
SUBJ_SHRINK = 0.1  # subject-mean shrinkage toward global


class ResponseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --- Sub 5 stack (encoder, MLP, Platt, T) ---
_bundle       = pickle.load(open("artifacts/bundle.pkl", "rb"))
_ENCODER_NAME = _bundle["encoder_name"]
_INPUT_DIM    = int(_bundle["input_dim"])

_ENCODER = SentenceTransformer(_ENCODER_NAME)
_X_MEAN  = np.load("artifacts/X_mean.npy")
_X_STD   = np.load("artifacts/X_std.npy") + 1e-8

_MLP = ResponseMLP(_INPUT_DIM)
_MLP.load_state_dict(torch.load("artifacts/mlp.pt", map_location="cpu", weights_only=True))
_MLP.eval()

_SUBJ_NAME_LKP  = json.load(open("artifacts/subject_name_lookup.json"))
_SUBJ_ID_LKP    = json.load(open("artifacts/subject_id_lookup.json"))
_SUBJ_EMB_INDEX = json.load(open("artifacts/subject_emb_index.json"))
_MEAN_THETA     = float(_SUBJ_NAME_LKP["__mean__"])
_V_SUBJECTS     = np.load("artifacts/subject_embeddings.npy")

_BM_EMB_LOOKUP   = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/bm_emb_lookup.json")).items()}
_COND_EMB_LOOKUP = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/cond_emb_lookup.json")).items()}

_SUBJ_MEAN_ACC   = json.load(open("artifacts/subject_mean_acc.json"))
_BM_MEAN_ACC     = json.load(open("artifacts/benchmark_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"])

_PLATT = json.load(open("artifacts/platt.json"))
_PLATT_A = float(_PLATT["a"])
_PLATT_B = float(_PLATT["b"])

try:
    _TEMPERATURE = float(json.load(open("artifacts/temperature.json"))["T"])
except (FileNotFoundError, KeyError, ValueError):
    _TEMPERATURE = 1.0


# --- kNN stack ---
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


# --- Caches ---
_item_text_cache: dict[str, np.ndarray] = {}  # raw MPNet embedding (768d)
_subject_cache:   dict[str, np.ndarray] = {}
_bm_cache:        dict[str, np.ndarray] = {}
_cond_cache:      dict[str, np.ndarray] = {}


def _extract_display_name(sc):
    first = (sc or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix): return first[len(prefix):].strip()
    return first.strip()


def _encode_item_raw(text):
    """Return 768d raw MPNet embedding, cached."""
    if text not in _item_text_cache:
        _item_text_cache[text] = _ENCODER.encode([text], convert_to_numpy=True)[0].astype(np.float32)
    return _item_text_cache[text]


def _encode_item_pca(text):
    """Return PCA-256 normalized embedding for kNN lookup."""
    raw = _encode_item_raw(text)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    return (proj / norm).astype(np.float32)


def _benchmark_emb(b):
    if b in _BM_EMB_LOOKUP: return _BM_EMB_LOOKUP[b]
    if b not in _bm_cache:
        _bm_cache[b] = _ENCODER.encode([f"Benchmark: {b}"], convert_to_numpy=True)[0].astype(np.float32)
    return _bm_cache[b]


def _condition_emb(c):
    c = c or "none"
    if c in _COND_EMB_LOOKUP: return _COND_EMB_LOOKUP[c]
    if c not in _cond_cache:
        _cond_cache[c] = _ENCODER.encode([f"Condition: {c}"], convert_to_numpy=True)[0].astype(np.float32)
    return _cond_cache[c]


def _lookup_theta(sc):
    name = _extract_display_name(sc)
    if name in _SUBJ_NAME_LKP: return float(_SUBJ_NAME_LKP[name])
    if name in _SUBJ_ID_LKP: return float(_SUBJ_ID_LKP[name])
    return _MEAN_THETA


def _lookup_subject_emb(sc):
    name = _extract_display_name(sc)
    if name in _SUBJ_EMB_INDEX: return _V_SUBJECTS[_SUBJ_EMB_INDEX[name]]
    if sc not in _subject_cache:
        _subject_cache[sc] = _ENCODER.encode([sc or ""], convert_to_numpy=True)[0].astype(np.float32)
    return _subject_cache[sc]


def _lookup_subject_mean_acc(sc):
    name = _extract_display_name(sc)
    if name in _SUBJ_MEAN_ACC: return float(_SUBJ_MEAN_ACC[name])
    return _GLOBAL_MEAN_ACC


def _subject_knn(name, query_emb):
    if name not in _PSR_INDEX: return None
    idx = int(_PSR_INDEX[name])
    start = int(_PSR_OFFSETS[idx]); end = int(_PSR_OFFSETS[idx + 1])
    if end - start == 0: return None
    item_rows = _PSR_ITEM_IDX[start:end]
    labels = _PSR_LABEL[start:end]
    sims = _ITEM_EMB_NORMED[item_rows] @ query_emb
    k = min(KNN_K, sims.shape[0])
    top_idx = np.argpartition(sims, -k)[-k:]
    top_sims = sims[top_idx]
    top_labels = labels[top_idx]
    weights = np.exp((top_sims - top_sims.max()) / KNN_TEMP)
    w_sum = float(weights.sum()) + 1e-8
    return float((weights * top_labels).sum() / w_sum)


def _build_x_sub5(theta, sc, item_emb_raw, b, c):
    s_emb = _lookup_subject_emb(sc)
    bm = _benchmark_emb(b)
    cd = _condition_emb(c)
    s_acc = _lookup_subject_mean_acc(sc)
    b_acc = float(_BM_MEAN_ACC.get(b, _GLOBAL_MEAN_ACC))
    x = np.concatenate([
        np.array([theta], dtype=np.float32),
        s_emb, item_emb_raw, bm, cd,
        np.array([s_acc, b_acc], dtype=np.float32),
    ]).astype(np.float32)
    return (x - _X_MEAN) / _X_STD


def predict(input, labeled=None):
    sc = input.get("subject_content", "")
    ic = input.get("item_content", "") or ""
    b = input.get("benchmark", "")
    c = input.get("condition", "")
    name = _extract_display_name(sc)

    # Smoothed subject-mean channel
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - SUBJ_SHRINK) * raw_subj + SUBJ_SHRINK * _GLOBAL_MEAN_ACC

    # Sub 5 (MLP + Platt + T) signal
    theta = _lookup_theta(sc)
    item_raw = _encode_item_raw(ic)
    x = _build_x_sub5(theta, sc, item_raw, b, c)
    with torch.no_grad():
        logit = float(_MLP(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item())
    z = (_PLATT_A * logit + _PLATT_B) / _TEMPERATURE
    p_sub5 = 1.0 / (1.0 + np.exp(-z))

    # kNN signal layered into subj_mean channel
    query_pca = _encode_item_pca(ic)
    p_knn = _subject_knn(name, query_pca)

    if p_knn is None:
        p_subj_channel = p_subj
    else:
        p_subj_channel = BETA * p_knn + (1.0 - BETA) * p_subj

    p_final = ALPHA * p_sub5 + (1.0 - ALPHA) * p_subj_channel
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
