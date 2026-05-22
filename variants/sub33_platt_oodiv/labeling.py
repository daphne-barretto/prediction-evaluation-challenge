"""labeling.py — out-of-distribution diversity acquisition for sub33_platt_oodiv.

Acquisition score = -cosine_similarity(item_emb, global_train_centroid),
where global_train_centroid is the L2-normalized mean of all PCA-256
training-item embeddings. Items whose text embedding is FURTHEST from
the training distribution centroid get the highest score, so within
each category the platform reveals the K=5 most "atypical" items.

Rationale (contrasted with uncertainty in sub33_platt_uncert):
* OOD items are exactly where sub 33's kNN is least reliable
  (cold-start by similarity, not by subject identity).
* If Platt + OOD-diversity beats Platt + uncertainty, the bottleneck
  was item-text coverage rather than calibration sharpness.
* If both informative acquisitions tie Platt + random, the K=5
  per-category budget is too small for acquisition policy to matter.

acquisition_function() never raises; on failure it falls back to a
constant 0.0 (which makes the platform use random fallback per the
sample labeling.py docstring).
"""
from __future__ import annotations

import json

import numpy as np


_LOADED = False
_ENCODER = None
_PCA_COMPONENTS_T = None
_CENTROID = None  # (256,) normalized mean of training item embeddings

_item_emb_cache: dict = {}


def _try_load() -> bool:
    global _LOADED, _PCA_COMPONENTS_T, _CENTROID
    if _LOADED:
        return True
    try:
        pca_components = np.load("artifacts/item_pca_components.npy").astype(np.float32)
        _PCA_COMPONENTS_T = pca_components.T

        item_emb_f16 = np.load("artifacts/item_embeddings_pca256_f16.npy")
        item_emb_f32 = item_emb_f16.astype(np.float32)
        norms = np.linalg.norm(item_emb_f32, axis=1, keepdims=True) + 1e-8
        item_emb_normed = item_emb_f32 / norms

        centroid = item_emb_normed.mean(axis=0)
        cnorm = float(np.linalg.norm(centroid)) + 1e-8
        _CENTROID = (centroid / cnorm).astype(np.float32)

        _LOADED = True
        return True
    except Exception:
        return False


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-mpnet-base-v2")
    return _ENCODER


def _encode_item(text: str):
    cache_key = text or ""
    if cache_key in _item_emb_cache:
        return _item_emb_cache[cache_key]
    raw = _get_encoder().encode([cache_key], convert_to_numpy=True)[0].astype(np.float32)
    proj = raw @ _PCA_COMPONENTS_T
    norm = float(np.linalg.norm(proj)) + 1e-8
    emb = (proj / norm).astype(np.float32)
    _item_emb_cache[cache_key] = emb
    return emb


def acquisition_function(input: dict) -> float:
    """Return -cosine_sim(item_emb, train_centroid). Higher = more OOD = more wanted."""
    if not _try_load():
        return 0.0
    try:
        emb = _encode_item(input.get("item_content", "") or "")
        sim = float(np.dot(emb, _CENTROID))
        return float(-sim)
    except Exception:
        return 0.0
