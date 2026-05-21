"""precompute_modal.py — One-off Modal job to compute artifacts needed for
the NLL-push variants beyond the -0.61 plateau.

Produces under /artifacts on the eval-artifacts volume:
  - item_embeddings_full.npy           (n_items × 768 float32)
  - item_id_lookup.json                (item_id → row idx)
  - item_mean_acc.json                 (item_id → mean accuracy, ≥3 responses)
  - subject_benchmark_mean_acc.json    ((s,b)-key → mean, ≥3 responses)
  - per_subject_responses.npz          (subject_id-indexed sparse responses)
  - item_diff_regressor.json           (W, b for item_emb → logit(μ_item))

Usage:
  MODAL_PROFILE=cs336-2026 modal run --detach precompute_modal.py
"""
from __future__ import annotations

import modal


app = modal.App("predictive-eval-precompute")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0", "numpy>=1.24", "pandas",
        "datasets>=2.14", "huggingface_hub",
        "sentence-transformers>=2.2", "scikit-learn>=1.3",
        "scipy", "tqdm",
    ])
)

volume = modal.Volume.from_name("eval-artifacts", create_if_missing=True)


ENCODER_NAME = "all-mpnet-base-v2"
ENCODER_DIM = 768


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    memory=65536,
    cpu=4.0,
    volumes={"/artifacts": volume},
)
def precompute():
    import json
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    REPO_ID = "aims-foundations/measurement-db"
    REGISTRY = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    OUT = "/artifacts"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load training data ─────────────────────────────────────────────────
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f for f in repo_files
        if f.endswith(".parquet")
        and f not in REGISTRY
        and not f.endswith("_traces.parquet")
    )
    features = Features({
        "subject_id":     Value("string"),
        "item_id":        Value("string"),
        "benchmark_id":   Value("string"),
        "trial":          Value("int64"),
        "test_condition": Value("string"),
        "response":       Value("float64"),
        "correct_answer": Value("string"),
        "trace":          Value("string"),
    })
    responses = load_dataset(REPO_ID, data_files=response_files,
                             features=features, split="train")
    items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
    print(f"Responses: {len(responses):,}  Items: {len(items):,}  Subjects: {len(subjects):,}")

    responses_df = responses.to_pandas()
    items_df = items.to_pandas()
    subjects_df = subjects.to_pandas()

    df = responses_df.merge(items_df[["item_id", "content"]], on="item_id", how="left")
    sub_cols = ["subject_id", "display_name"]
    df = df.merge(subjects_df[sub_cols], on="subject_id", how="left")
    df["benchmark"] = df["benchmark_id"]
    df["condition"] = df["test_condition"].fillna("none").replace("", "none")
    df["item_content"] = df["content"]

    def binarize(row):
        v, bm = row["response"], str(row["benchmark"])
        if bm == "mtbench":       return 1.0 if v >= 5.0 else 0.0
        if bm == "ultrafeedback": return 1.0 if v >= 4.0 else 0.0
        if bm == "rewardbench":   return 1.0 if v >= 0.5 else 0.0
        if bm in ("cybench", "livecodebench", "matharena"):
            return 1.0 if v > 0.0 else 0.0
        return 1.0 if v >= 0.5 else 0.0

    df["label"] = df.apply(binarize, axis=1)
    train_df = df[["subject_id", "item_id", "benchmark", "condition",
                   "display_name", "item_content", "label"]].copy()
    train_df["label"] = train_df["label"].astype(np.float32)
    print(f"train_df: {len(train_df):,} rows  pass_rate={train_df['label'].mean():.4f}")

    # ── 2. Item embeddings (FULL set) ─────────────────────────────────────────
    item_text_map = train_df.drop_duplicates("item_id").set_index("item_id")["item_content"]
    item_ids = list(item_text_map.index)
    item_texts = [str(item_text_map[iid]) for iid in item_ids]
    n_items = len(item_ids)
    print(f"\nEncoding {n_items:,} unique items...")

    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    V_items = encoder.encode(item_texts, batch_size=128,
                             show_progress_bar=True,
                             convert_to_numpy=True).astype(np.float32)
    print(f"item_embeddings shape: {V_items.shape}")
    np.save(f"{OUT}/item_embeddings_full.npy", V_items)
    with open(f"{OUT}/item_id_lookup.json", "w") as f:
        json.dump({iid: i for i, iid in enumerate(item_ids)}, f)
    print("  saved item_embeddings_full.npy + item_id_lookup.json")

    # ── 3. Item mean accuracy ─────────────────────────────────────────────────
    item_counts = train_df.groupby("item_id")["label"].count()
    item_means = train_df.groupby("item_id")["label"].mean()
    global_mean = float(train_df["label"].mean())
    n_prior = 5.0
    smoothed_item_mean = (
        (item_means * item_counts + global_mean * n_prior) / (item_counts + n_prior)
    )
    item_mean_dict = {iid: float(smoothed_item_mean[iid]) for iid in item_ids
                      if iid in smoothed_item_mean.index}
    with open(f"{OUT}/item_mean_acc.json", "w") as f:
        json.dump({"global_mean": global_mean, "n_prior": n_prior,
                   "items": item_mean_dict}, f)
    print(f"  item_mean: {len(item_mean_dict):,} items  global={global_mean:.4f}")

    # ── 4. Subject × Benchmark cross-table ────────────────────────────────────
    sb_counts = train_df.groupby(["display_name", "benchmark"])["label"].count()
    sb_means = train_df.groupby(["display_name", "benchmark"])["label"].mean()
    s_counts = train_df.groupby("display_name")["label"].count()
    s_means = train_df.groupby("display_name")["label"].mean()
    n_prior_sb = 3.0
    sb_dict: dict[str, float] = {}
    for (s, b), n in sb_counts.items():
        smoothed = (sb_means[(s, b)] * n + s_means[s] * n_prior_sb) / (n + n_prior_sb)
        sb_dict[f"{s}|||{b}"] = float(smoothed)
    with open(f"{OUT}/subject_benchmark_mean_acc.json", "w") as f:
        json.dump({"sep": "|||", "global_mean": global_mean,
                   "subject_smoothing": n_prior_sb,
                   "pairs": sb_dict}, f)
    print(f"  (s,b) pairs: {len(sb_dict):,}")

    # ── 5. Per-subject sparse response index ──────────────────────────────────
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    subjects_unique = sorted(train_df["display_name"].dropna().unique())
    subject_to_offset = {}
    item_idx_arr: list[int] = []
    label_arr: list[float] = []
    offsets = [0]
    for s in subjects_unique:
        sub = train_df[train_df["display_name"] == s][["item_id", "label"]]
        for iid, lab in zip(sub["item_id"].values, sub["label"].values):
            if iid in item_id_to_idx:
                item_idx_arr.append(item_id_to_idx[iid])
                label_arr.append(float(lab))
        offsets.append(len(item_idx_arr))
        subject_to_offset[s] = len(offsets) - 2
    item_idx_arr_np = np.asarray(item_idx_arr, dtype=np.int32)
    label_arr_np = np.asarray(label_arr, dtype=np.float32)
    offsets_np = np.asarray(offsets, dtype=np.int64)
    np.savez(f"{OUT}/per_subject_responses.npz",
             item_idx=item_idx_arr_np,
             label=label_arr_np,
             offsets=offsets_np,
             subjects=np.array(subjects_unique, dtype=object))
    with open(f"{OUT}/per_subject_responses_index.json", "w") as f:
        json.dump(subject_to_offset, f)
    print(f"  per_subject: {len(subjects_unique):,} subjects, "
          f"{len(item_idx_arr_np):,} (s,i,y) tuples")

    # ── 6. Item-difficulty regressor (linear: emb → logit μ_item) ─────────────
    from sklearn.linear_model import Ridge
    eps = 1e-3
    y_logit = np.log(np.clip(smoothed_item_mean.values, eps, 1 - eps)
                     / np.clip(1 - smoothed_item_mean.values, eps, 1 - eps))
    y_logit = y_logit.astype(np.float32)
    train_items = list(smoothed_item_mean.index)
    train_item_rows = np.asarray([item_id_to_idx[iid] for iid in train_items])
    X_diff = V_items[train_item_rows]
    print(f"  Fitting Ridge: X={X_diff.shape}, y={y_logit.shape}")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_diff, y_logit)
    pred = ridge.predict(X_diff)
    rmse = float(np.sqrt(np.mean((pred - y_logit) ** 2)))
    corr = float(np.corrcoef(pred, y_logit)[0, 1])
    print(f"  Ridge train RMSE={rmse:.4f}  corr={corr:.4f}")
    with open(f"{OUT}/item_diff_regressor.json", "w") as f:
        json.dump({
            "coef": ridge.coef_.astype(np.float32).tolist(),
            "intercept": float(ridge.intercept_),
            "global_logit": float(np.log(global_mean / (1 - global_mean))),
            "train_rmse": rmse,
            "train_corr": corr,
        }, f)
    print("  saved item_diff_regressor.json")

    volume.commit()
    print("\nAll artifacts committed to volume.")


@app.local_entrypoint()
def main():
    print("Starting precompute on Modal GPU (detached)...")
    precompute.remote()
    print("Done — artifacts on eval-artifacts volume.")
