"""item_irt_2pl_nn (Prong-F2) — 2PL IRT prediction with 1-NN item lookup.

Closes the F2 (latent factor) cell of the coverage table.

Setup:
- theta[s] is the trained 2PL ability for each of 909 training subjects.
- item_a[i], item_b[i] are the trained 2PL discrimination/difficulty for each
  of 103983 training items.

At test time:
- Look up theta for subject by display name (subject_name_lookup.json maps
  name -> theta value directly; we use it as θ_s).
- Encode the test item with MPNet, find the 1 nearest training item by
  cosine sim on PCA-256 MPNet embeddings, and reuse that item's (a, b).
- a is clipped to [0, A_MAX] because some training items have saturated
  discrimination (max raw a ~ 1682) that would push σ to 0/1.
- Predict p = σ(a * (θ_s - b)), blended with the smoothed subject mean.
- Cold-start fallback: smoothed subject_mean for unknown subjects.

This is functionally an IRT analogue of sub 33 (item-text k-NN within
subject): sub 33 weights raw labels of similar items, this maps item text
to latent (a, b) params and predicts via the 2PL likelihood.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer


ALPHA = 0.1   # subj-mean shrinkage toward global
BETA = 0.4    # blend weight on IRT prediction vs smoothed subject mean
A_MAX = 4.0   # clip item discrimination (raw max is ~1682, mean ~5)


_ENCODER_NAME = "all-mpnet-base-v2"
_ENCODER = None


_SUBJ_MEAN_ACC = json.load(open("artifacts/subject_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)
_SUBJECT_NAME_LOOKUP = json.load(open("artifacts/subject_name_lookup.json"))
_ITEM_A = np.clip(np.load("artifacts/item_a.npy"), 0.0, A_MAX).astype(np.float32)
_ITEM_B = np.load("artifacts/item_b.npy").astype(np.float32)

_ITEM_EMB_F16 = np.load("artifacts/item_embeddings_pca256_f16.npy")
_PCA_COMPONENTS = np.load("artifacts/item_pca_components.npy").astype(np.float32)
_PCA_COMPONENTS_T = _PCA_COMPONENTS.T
_ITEM_EMB_F32 = _ITEM_EMB_F16.astype(np.float32)
_ITEM_NORMS = np.linalg.norm(_ITEM_EMB_F32, axis=1, keepdims=True) + 1e-8
_ITEM_EMB_NORMED = (_ITEM_EMB_F32 / _ITEM_NORMS).astype(np.float32)

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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    name = _extract_display_name(input.get("subject_content", ""))
    theta_s = _SUBJECT_NAME_LOOKUP.get(name, None)
    raw_subj = float(_SUBJ_MEAN_ACC.get(name, _GLOBAL_MEAN_ACC))
    p_subj = (1.0 - ALPHA) * raw_subj + ALPHA * _GLOBAL_MEAN_ACC

    if theta_s is None:
        return float(np.clip(p_subj, 1e-3, 1 - 1e-3))

    item_content = input.get("item_content", "") or ""
    emb = _encode_item(item_content)
    sims = _ITEM_EMB_NORMED @ emb
    nn = int(np.argmax(sims))

    a = float(_ITEM_A[nn])
    b = float(_ITEM_B[nn])
    p_irt = _sigmoid(a * (float(theta_s) - b))
    p_final = BETA * p_irt + (1.0 - BETA) * p_subj
    return float(np.clip(p_final, 1e-3, 1 - 1e-3))
