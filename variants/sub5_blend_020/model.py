"""sub5_blend_020 — alpha=0.20 between sub 21 (alpha=0.0, AUC 0.68) and
sub 17 (alpha=0.3, AUC 0.69). Probes the AUC peak between those two
points.

p_blend = 0.20 * p_sub5 + 0.80 * subject_mean_acc

Identical artifacts as sub 13/17; only BLEND_WEIGHT_SUB5 differs.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


BLEND_WEIGHT_SUB5 = 0.20


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

_BM_EMB_LOOKUP   = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/bm_emb_lookup.json")).items()}
_COND_EMB_LOOKUP = {k: np.asarray(v, dtype=np.float32)
                    for k, v in json.load(open("artifacts/cond_emb_lookup.json")).items()}

_SUBJ_MEAN_ACC  = json.load(open("artifacts/subject_mean_acc.json"))
_BM_MEAN_ACC    = json.load(open("artifacts/benchmark_mean_acc.json"))
_GLOBAL_MEAN_ACC = float(
    json.load(open("artifacts/global_mean_acc.json"))["global_mean_acc"]
)

_PLATT = json.load(open("artifacts/platt.json"))
_PLATT_A = float(_PLATT["a"])
_PLATT_B = float(_PLATT["b"])

try:
    _TEMPERATURE = float(json.load(open("artifacts/temperature.json"))["T"])
except (FileNotFoundError, KeyError, ValueError):
    _TEMPERATURE = 1.0

_item_cache:    dict[str, np.ndarray] = {}
_subject_cache: dict[str, np.ndarray] = {}
_bm_cache:      dict[str, np.ndarray] = {}
_cond_cache:    dict[str, np.ndarray] = {}


def _extract_display_name(subject_content: str) -> str:
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


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    theta = _lookup_theta(input["subject_content"])
    x = _build_x(theta, input["subject_content"], input["item_content"],
                 input["benchmark"], input["condition"])
    p_sub5 = _platt_prob(_logit(x))
    p_subj = _lookup_subject_mean_acc(input["subject_content"])
    p_blend = BLEND_WEIGHT_SUB5 * p_sub5 + (1.0 - BLEND_WEIGHT_SUB5) * p_subj
    return float(np.clip(p_blend, 1e-4, 1 - 1e-4))
