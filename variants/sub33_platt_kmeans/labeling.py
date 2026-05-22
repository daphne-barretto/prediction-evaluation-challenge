"""labeling.py — k-means diversity acquisition for sub33_platt_kmeans.

Implements the worked-example §3.6 design literally:
  * Offline KMeans (n_clusters=64) on training-item PCA-256 embeddings.
  * Module-level Counter `_seen` tracks per-cluster visit counts within
    this round's iteration through the candidate pool.
  * acquisition_function(input):
        c = nearest_centroid(test_item_emb)
        _seen[c] += 1
        return -_seen[c]
    First-of-its-cluster scores -1 (highest); duplicates within a cluster
    score progressively lower, so the platform's top-K=5 picks span
    distinct clusters as long as the pool contains >=5 distinct cluster
    members.

Differences from sub33_platt_oodiv (which uses a single training centroid):
  * That sibling scored by similarity to the GLOBAL training mean (a
    cold-start-by-distance signal, but with no cross-candidate state).
  * This variant uses 64 clusters and within-round visit counts, so the
    K=5 budget is forced to cover 5 distinct semantic neighborhoods.

KMeans is fit at module load with random_state=0 so the centroids are
deterministic across container restarts.

acquisition_function() never raises; on failure it returns a constant
0.0, which the platform treats identically to a missing labeling.py
(random fallback).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


_LOADED = False
_ENCODER = None
_PCA_COMPONENTS_T = None
_CENTROIDS = None       # (n_clusters, 256) L2-normalized
_N_CLUSTERS = 64

_item_emb_cache: dict = {}
_seen: Counter = Counter()


def _try_load() -> bool:
    global _LOADED, _PCA_COMPONENTS_T, _CENTROIDS
    if _LOADED:
        return True
    try:
        from sklearn.cluster import KMeans  # type: ignore

        pca_components = np.load("artifacts/item_pca_components.npy").astype(np.float32)
        _PCA_COMPONENTS_T = pca_components.T

        item_emb_f16 = np.load("artifacts/item_embeddings_pca256_f16.npy")
        item_emb_f32 = item_emb_f16.astype(np.float32)
        norms = np.linalg.norm(item_emb_f32, axis=1, keepdims=True) + 1e-8
        item_emb_normed = item_emb_f32 / norms

        km = KMeans(n_clusters=_N_CLUSTERS, n_init=10, random_state=0)
        km.fit(item_emb_normed)
        centroids = km.cluster_centers_.astype(np.float32)
        cnorms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
        _CENTROIDS = (centroids / cnorms).astype(np.float32)

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


def acquisition_function(input: dict) -> float:
    """k-means diversity: -visits_to_nearest_centroid (so far this round).

    First-of-its-cluster -> -1 (highest score).
    Duplicates within a cluster score -2, -3, ...
    """
    if not _try_load():
        return 0.0
    try:
        emb = _encode_item(input.get("item_content", "") or "")
        # Nearest centroid by euclidean distance (== smallest -cos on unit sphere).
        d = np.linalg.norm(_CENTROIDS - emb[None, :], axis=1)
        c = int(np.argmin(d))
        _seen[c] += 1
        return float(-_seen[c])
    except Exception:
        return 0.0
