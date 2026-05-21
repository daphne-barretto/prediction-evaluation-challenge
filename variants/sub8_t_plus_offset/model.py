"""
model.py — Predict entry point for the Predictive AI Evaluation Challenge.

Feature vector (input_dim = 3075):
  [theta(1) | subject_emb(768) | item_emb(768) |
   benchmark_emb(768) | condition_emb(768) |
   subject_mean_acc(1) | benchmark_mean_acc(1)]

Inference pipeline:
  1. Look up theta + subject_emb + mean accs by display_name (fallback: encode
     subject text fresh / global mean).
  2. Encode item, benchmark, condition text via all-mpnet-base-v2 (cached).
  3. Normalize features and forward through the MLP to get a logit.
  4. Apply Platt calibration p = sigmoid(a * logit + b).
  5. If `labeled` items are provided, compute the mean residual (predicted -
     actual) per subject from labeled, and shift the prediction for the same
     subject by that offset.
"""
from __future__ import annotations

import json
import pickle
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


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


# ── Load once at container start ──────────────────────────────────────────────
_bundle        = pickle.load(open("artifacts/bundle.pkl", "rb"))
_ENCODER_NAME  = _bundle["encoder_name"]
_ENCODER_DIM   = int(_bundle["encoder_dim"])
_INPUT_DIM     = int(_bundle["input_dim"])

_ENCODER = SentenceTransformer(_ENCODER_NAME)
_X_MEAN  = np.load("artifacts/X_mean.npy")
_X_STD   = np.load("artifacts/X_std.npy") + 1e-8

_MLP = ResponseMLP(_INPUT_DIM)
_MLP.load_state_dict(torch.load("artifacts/mlp.pt", map_location="cpu",
                                 weights_only=True))
_MLP.eval()

_SUBJ_NAME_LKP  = json.load(open("artifacts/subject_name_lookup.json"))
_SUBJ_ID_LKP    = json.load(open("artifacts/subject_id_lookup.json"))
_SUBJ_EMB_INDEX = json.load(open("artifacts/subject_emb_index.json"))
_MEAN_THETA     = float(_SUBJ_NAME_LKP["__mean__"])
_V_SUBJECTS     = np.load("artifacts/subject_embeddings.npy")
_MEAN_SUBJ_EMB  = _V_SUBJECTS.mean(axis=0)

# Benchmark / condition text-embedding lookups (cold-start: encode on the fly).
_BM_EMB_LOOKUP   = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/bm_emb_lookup.json")).items()}
_COND_EMB_LOOKUP = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/cond_emb_lookup.json")).items()}

_SUBJ_MEAN_ACC  = json.load(open("artifacts/subject_mean_acc.json"))
_BM_MEAN_ACC    = json.load(open("artifacts/benchmark_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"])

_PLATT = json.load(open("artifacts/platt.json"))
_PLATT_A = float(_PLATT["a"])
_PLATT_B = float(_PLATT["b"])

try:
    _TEMPERATURE = float(json.load(open("artifacts/temperature.json"))["T"])
except (FileNotFoundError, KeyError, ValueError):
    _TEMPERATURE = 1.0

# Per-round caches
_item_cache:    dict[str, np.ndarray] = {}
_subject_cache: dict[str, np.ndarray] = {}
_bm_cache:      dict[str, np.ndarray] = {}
_cond_cache:    dict[str, np.ndarray] = {}


def _extract_display_name(subject_content: str) -> str:
    """Strip a leading 'Name: ' / 'display_name: ' / etc. from the first line."""
    first = (subject_content or "").split("\n", 1)[0]
    for prefix in ("Name: ", "display_name: ", "Display Name: ", "name: "):
        if first.startswith(prefix):
            return first[len(prefix):].strip()
    return first.strip()


def _encode_text(text: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if text not in cache:
        cache[text] = _ENCODER.encode([text], convert_to_numpy=True)[0].astype(np.float32)
    return cache[text]


def _encode_item(text: str) -> np.ndarray:
    return _encode_text(text or "", _item_cache)


def _encode_subject(text: str) -> np.ndarray:
    return _encode_text(text or "", _subject_cache)


def _benchmark_emb(benchmark: str) -> np.ndarray:
    if benchmark in _BM_EMB_LOOKUP:
        return _BM_EMB_LOOKUP[benchmark]
    if benchmark not in _bm_cache:
        _bm_cache[benchmark] = _ENCODER.encode(
            [f"Benchmark: {benchmark}"], convert_to_numpy=True
        )[0].astype(np.float32)
    return _bm_cache[benchmark]


def _condition_emb(condition: str) -> np.ndarray:
    cond = condition or "none"
    if cond in _COND_EMB_LOOKUP:
        return _COND_EMB_LOOKUP[cond]
    if cond not in _cond_cache:
        _cond_cache[cond] = _ENCODER.encode(
            [f"Condition: {cond}"], convert_to_numpy=True
        )[0].astype(np.float32)
    return _cond_cache[cond]


def _lookup_theta(subject_content: str) -> float:
    name = _extract_display_name(subject_content)
    if name in _SUBJ_NAME_LKP:
        return float(_SUBJ_NAME_LKP[name])
    if name in _SUBJ_ID_LKP:
        return float(_SUBJ_ID_LKP[name])
    return _MEAN_THETA


def _lookup_subject_emb(subject_content: str) -> np.ndarray:
    name = _extract_display_name(subject_content)
    if name in _SUBJ_EMB_INDEX:
        return _V_SUBJECTS[_SUBJ_EMB_INDEX[name]]
    return _encode_subject(subject_content)


def _lookup_subject_mean_acc(subject_content: str) -> float:
    name = _extract_display_name(subject_content)
    if name in _SUBJ_MEAN_ACC:
        return float(_SUBJ_MEAN_ACC[name])
    return _GLOBAL_MEAN_ACC


def _lookup_benchmark_mean_acc(benchmark: str) -> float:
    if benchmark in _BM_MEAN_ACC:
        return float(_BM_MEAN_ACC[benchmark])
    return _GLOBAL_MEAN_ACC


def _build_x(theta: float, subject_content: str, item_content: str,
             benchmark: str, condition: str) -> np.ndarray:
    s_emb  = _lookup_subject_emb(subject_content)
    i_emb  = _encode_item(item_content)
    bm_emb = _benchmark_emb(benchmark)
    cd_emb = _condition_emb(condition)
    s_acc  = _lookup_subject_mean_acc(subject_content)
    b_acc  = _lookup_benchmark_mean_acc(benchmark)
    x = np.concatenate([
        np.array([theta], dtype=np.float32),
        s_emb, i_emb, bm_emb, cd_emb,
        np.array([s_acc, b_acc], dtype=np.float32),
    ]).astype(np.float32)
    if x.shape[0] != _INPUT_DIM:
        raise ValueError(
            f"Feature dim mismatch: got {x.shape[0]}, expected {_INPUT_DIM}"
        )
    return (x - _X_MEAN) / _X_STD


def _logit(x: np.ndarray) -> float:
    with torch.no_grad():
        return float(_MLP(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item())


def _platt_prob(logit: float) -> float:
    z = (_PLATT_A * logit + _PLATT_B) / _TEMPERATURE
    return float(1.0 / (1.0 + np.exp(-z)))


# ── Per-subject mean-residual offset (replaces Newton-Raphson) ────────────────
_offset_cache: dict[int, dict[str, float]] = {}


def _compute_subject_offsets(labeled: Iterable[dict]) -> dict[str, float]:
    """For each subject in `labeled`, return mean(pred - actual)."""
    by_subject: dict[str, list[float]] = {}
    for ex in labeled:
        try:
            theta = _lookup_theta(ex["subject_content"])
            x     = _build_x(theta, ex["subject_content"], ex["item_content"],
                             ex["benchmark"], ex["condition"])
            p     = _platt_prob(_logit(x))
            y     = float(ex["label"])
            name  = _extract_display_name(ex["subject_content"])
            by_subject.setdefault(name, []).append(p - y)
        except Exception:
            continue
    return {name: float(np.mean(rs)) for name, rs in by_subject.items() if rs}


def _get_subject_offsets(labeled: list[dict] | None) -> dict[str, float]:
    if not labeled:
        return {}
    key = id(labeled)
    if key not in _offset_cache:
        _offset_cache[key] = _compute_subject_offsets(labeled)
    return _offset_cache[key]


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    # sub 8 variant: T-scaling AND per-subject mean-residual offset.
    # Hypothesis: in sub 2 the offset hurt because raw predictions were
    # wildly overconfident (mean residuals were dominated by sigmoid
    # saturation rather than per-subject bias). With T-scaling
    # softening predictions to well-calibrated probabilities, the
    # mean-residual offset may now correctly capture per-subject bias.
    theta = _lookup_theta(input["subject_content"])
    x = _build_x(theta, input["subject_content"], input["item_content"],
                 input["benchmark"], input["condition"])
    p = _platt_prob(_logit(x))

    offsets = _get_subject_offsets(labeled)
    if offsets:
        name = _extract_display_name(input["subject_content"])
        if name in offsets:
            p = p - offsets[name]

    return float(np.clip(p, 1e-4, 1 - 1e-4))
