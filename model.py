"""
model.py — Improved entry point for the Predictive AI Evaluation Challenge.

Features: [theta | subject_emb | item_emb | benchmark_ohe | condition_ohe]
"""
from __future__ import annotations
import json
import pickle
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
_BM_COLS       = _bundle["bm_columns"]
_COND_COLS     = _bundle["cond_columns"]
_ENCODER_NAME  = _bundle["encoder_name"]
_INPUT_DIM     = _bundle["input_dim"]

_ENCODER       = SentenceTransformer(_ENCODER_NAME)
_X_MEAN        = np.load("artifacts/X_mean.npy")
_X_STD         = np.load("artifacts/X_std.npy") + 1e-8

_MLP           = ResponseMLP(_INPUT_DIM)
_MLP.load_state_dict(torch.load("artifacts/mlp.pt", map_location="cpu",
                                 weights_only=True))
_MLP.eval()

_SUBJ_NAME_LKP  = json.load(open("artifacts/subject_name_lookup.json"))
_SUBJ_ID_LKP    = json.load(open("artifacts/subject_id_lookup.json"))
_SUBJ_EMB_INDEX = json.load(open("artifacts/subject_emb_index.json"))
_MEAN_THETA     = float(_SUBJ_NAME_LKP["__mean__"])

# Load subject embeddings for lookup
_V_SUBJECTS = np.load("artifacts/subject_embeddings.npy")
_MEAN_SUBJ_EMB = _V_SUBJECTS.mean(axis=0)

# Per-round caches
_item_cache:    dict[str, np.ndarray] = {}
_subject_cache: dict[str, np.ndarray] = {}


def _encode_item(text: str) -> np.ndarray:
    if text not in _item_cache:
        _item_cache[text] = _ENCODER.encode([text], convert_to_numpy=True)[0]
    return _item_cache[text]


def _encode_subject(subject_content: str) -> np.ndarray:
    if subject_content not in _subject_cache:
        _subject_cache[subject_content] = _ENCODER.encode(
            [subject_content], convert_to_numpy=True)[0]
    return _subject_cache[subject_content]


def _lookup_theta(subject_content: str) -> float:
    display_name = subject_content.split("\n")[0].replace("Name: ", "").strip()
    if display_name in _SUBJ_NAME_LKP:
        return float(_SUBJ_NAME_LKP[display_name])
    if display_name in _SUBJ_ID_LKP:
        return float(_SUBJ_ID_LKP[display_name])
    return _MEAN_THETA


def _lookup_subject_emb(subject_content: str) -> np.ndarray:
    """Get subject embedding — from lookup if known, else encode fresh."""
    display_name = subject_content.split("\n")[0].replace("Name: ", "").strip()
    if display_name in _SUBJ_EMB_INDEX:
        idx = _SUBJ_EMB_INDEX[display_name]
        return _V_SUBJECTS[idx]
    # Unknown subject — encode its text directly
    return _encode_subject(subject_content)


def _build_x(theta: float, subject_content: str,
              item_content: str, benchmark: str, condition: str) -> np.ndarray:
    s_emb = _lookup_subject_emb(subject_content)
    i_emb = _encode_item(item_content)
    bm_v  = np.zeros(len(_BM_COLS),   dtype=np.float32)
    cd_v  = np.zeros(len(_COND_COLS), dtype=np.float32)
    bm_key = f"bm_{benchmark}"
    cd_key = f"cond_{condition}"
    if bm_key in _BM_COLS:   bm_v[_BM_COLS.index(bm_key)]   = 1.0
    if cd_key in _COND_COLS: cd_v[_COND_COLS.index(cd_key)] = 1.0
    x = np.hstack([[theta], s_emb, i_emb, bm_v, cd_v]).astype(np.float32)
    return (x - _X_MEAN) / _X_STD


def _calibrate_theta(theta_prior: float, labeled: list[dict],
                     n_steps: int = 10, reg: float = 0.1) -> float:
    theta = theta_prior
    for _ in range(n_steps):
        grad = -reg * (theta - theta_prior)
        hess = -reg
        for ex in labeled:
            x = _build_x(theta, ex["subject_content"], ex["item_content"],
                         ex["benchmark"], ex["condition"])
            with torch.no_grad():
                logit = _MLP(torch.tensor(x).unsqueeze(0)).item()
            p     = float(torch.sigmoid(torch.tensor(logit)).item())
            y     = int(ex["label"])
            grad += y - p
            hess -= p * (1.0 - p)
        if abs(hess) < 1e-8:
            break
        step   = grad / hess
        theta -= step
        if abs(step) < 1e-6:
            break
    return float(theta)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    theta = _lookup_theta(input["subject_content"])
    if labeled:
        theta = _calibrate_theta(theta, labeled)
    x = _build_x(theta, input["subject_content"], input["item_content"],
                 input["benchmark"], input["condition"])
    with torch.no_grad():
        logit = _MLP(torch.tensor(x).unsqueeze(0)).item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    return float(np.clip(prob, 1e-4, 1 - 1e-4))