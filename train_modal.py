"""
train_modal.py — Full training pipeline on Modal GPU.

Pipeline:
  1. Load HF response/items/subjects/benchmarks tables.
  2. Binarize labels per benchmark convention.
  3. Fit 2PL IRT on the dense response sub-matrix to get theta per subject.
  4. Encode subjects, items, benchmark names, and condition names with
     all-mpnet-base-v2 (768d).
  5. Build feature matrix:
        [theta(1) | subject_emb(768) | item_emb(768) |
         benchmark_emb(768) | condition_emb(768) |
         subject_mean_acc(1) | benchmark_mean_acc(1)]
     (input_dim = 3075)
  6. Cold-start validation: hold out 2 entire benchmarks, evaluate
     calibrated NLL on those rows after training on the rest.
  7. Platt calibration: hold out 10% of remaining rows, fit a*logit + b
     via sklearn LogisticRegression on logits.

Usage:
    modal run train_modal.py
"""

import modal
import os

app = modal.App("predictive-eval-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git"])
    .pip_install([
        "torch>=2.0", "numpy>=1.24", "pandas",
        "datasets>=2.14", "huggingface_hub",
        "sentence-transformers>=2.2", "scikit-learn>=1.3",
        "scipy", "transformers>=4.30", "tqdm",
    ])
    .run_commands(
        "git clone https://github.com/aims-foundations/torch_measure.git /tmp/torch_measure",
        "pip install -e '/tmp/torch_measure[all]'",
    )
)

volume = modal.Volume.from_name("eval-artifacts", create_if_missing=True)


ENCODER_NAME = "all-mpnet-base-v2"
ENCODER_DIM  = 768
COLD_START_HOLDOUT_BENCHMARKS = 2   # number of entire benchmarks held out for cold-start eval
PLATT_HOLDOUT_FRAC            = 0.10


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    memory=65536,
    cpu=4.0,
    volumes={"/artifacts": volume},
)
def run_training():
    import sys
    sys.path.insert(0, "/tmp/torch_measure/src")

    import os, json, pickle, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LogisticRegression
    from torch_measure.models import TwoPL

    REPO_ID        = "aims-foundations/measurement-db"
    REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    ARTIFACTS_DIR  = "/artifacts"
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("Discovering response tables...")
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )
    response_features = Features({
        "subject_id":     Value("string"),
        "item_id":        Value("string"),
        "benchmark_id":   Value("string"),
        "trial":          Value("int64"),
        "test_condition": Value("string"),
        "response":       Value("float64"),
        "correct_answer": Value("string"),
        "trace":          Value("string"),
    })
    responses  = load_dataset(REPO_ID, data_files=response_files,
                              features=response_features, split="train")
    items      = load_dataset(REPO_ID, data_files="items.parquet",      split="train")
    subjects   = load_dataset(REPO_ID, data_files="subjects.parquet",   split="train")
    benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")
    print(f"Responses: {len(responses):,}  Items: {len(items):,}  "
          f"Subjects: {len(subjects):,}  Benchmarks: {len(benchmarks):,}")

    # ── 2. Build train_df ──────────────────────────────────────────────────────
    responses_df  = responses.to_pandas()
    items_df      = items.to_pandas()
    subjects_df   = subjects.to_pandas()
    benchmarks_df = benchmarks.to_pandas()

    df = responses_df.merge(items_df[["item_id", "content"]], on="item_id", how="left")
    df = df.merge(
        subjects_df[["subject_id", "display_name", "provider", "params",
                     "release_date", "family"]]
        if "family" in subjects_df.columns
        else subjects_df[["subject_id", "display_name", "provider",
                           "params", "release_date"]],
        on="subject_id", how="left"
    )
    df = df.merge(
        benchmarks_df[["benchmark_id", "name", "domain", "modality", "response_type"]],
        on="benchmark_id", how="left"
    )
    df["benchmark"]    = df["benchmark_id"]
    df["condition"]    = df["test_condition"].fillna("none").replace("", "none")
    df["item_content"] = df["content"]
    df["subject_content"] = (
        "Name: "        + df["display_name"].fillna(df["subject_id"]) +
        "\nProvider: "  + df["provider"].fillna("") +
        "\nParameters: "+ df["params"].fillna("").astype(str) +
        "\nReleased: "  + df["release_date"].fillna("").astype(str)
    )

    def binarize(row):
        v, bm = row["response"], str(row["benchmark"])
        if bm == "mtbench":       return 1.0 if v >= 5.0 else 0.0
        if bm == "ultrafeedback": return 1.0 if v >= 4.0 else 0.0
        if bm == "rewardbench":   return 1.0 if v >= 0.5 else 0.0
        if bm in ("cybench", "livecodebench", "matharena"):
            return 1.0 if v > 0.0 else 0.0
        return 1.0 if v >= 0.5 else 0.0

    print("Binarizing labels...")
    df["label"] = df.apply(binarize, axis=1)
    train_df = df[["subject_id", "item_id", "benchmark", "condition",
                   "subject_content", "item_content", "label"]].copy()
    print(f"train_df: {len(train_df):,} rows  pass_rate={train_df['label'].mean():.3f}")

    # ── 3. Build response matrix ───────────────────────────────────────────────
    print("\nBuilding response matrix...")
    subject_ids   = train_df["subject_id"].unique()
    item_ids      = train_df["item_id"].unique()
    subject_index = {sid: i for i, sid in enumerate(subject_ids)}
    item_index    = {iid: j for j, iid in enumerate(item_ids)}
    n_subjects, n_items = len(subject_ids), len(item_ids)

    R = np.full((n_subjects, n_items), np.nan)
    for row in train_df.itertuples(index=False):
        R[subject_index[row.subject_id], item_index[row.item_id]] = float(row.label)

    dense_indices = np.where((~np.isnan(R)).sum(axis=0) >= 5)[0]
    R_dense       = R[:, dense_indices]
    print(f"Dense items: {len(dense_indices):,} / {n_items:,}")

    item_id_ordered    = sorted(item_index,    key=lambda x: item_index[x])
    subject_id_ordered = sorted(subject_index, key=lambda x: subject_index[x])
    item_text_map    = train_df.drop_duplicates("item_id").set_index("item_id")["item_content"]
    subject_text_map = train_df.drop_duplicates("subject_id").set_index("subject_id")["subject_content"]
    item_meta_map    = (train_df.drop_duplicates("item_id")
                        .set_index("item_id")[["benchmark", "condition"]].astype(str))
    item_texts       = [str(item_text_map[iid])    for iid in item_id_ordered]
    subject_texts    = [str(subject_text_map[sid]) for sid in subject_id_ordered]
    dense_item_ids   = [item_id_ordered[j]         for j   in dense_indices]
    item_texts_dense = [item_texts[j]              for j   in dense_indices]
    dense_benchmarks = [item_meta_map.loc[iid, "benchmark"] if iid in item_meta_map.index
                        else "unknown" for iid in dense_item_ids]
    dense_conditions = [item_meta_map.loc[iid, "condition"] if iid in item_meta_map.index
                        else "none"    for iid in dense_item_ids]

    # ── 4. Fit 2PL IRT ─────────────────────────────────────────────────────────
    print(f"\nFitting 2PL IRT: {n_subjects} subjects x {len(dense_indices)} dense items")
    data_t = torch.tensor(R_dense, dtype=torch.float32)
    mask_t = ~torch.isnan(data_t)
    data_t = torch.nan_to_num(data_t, nan=0.0)
    twopl   = TwoPL(n_subjects=n_subjects, n_items=len(dense_indices), device=device)
    history = twopl.fit(data_t.to(device), mask=mask_t.to(device),
                        max_epochs=1000, verbose=True)
    print(f"IRT loss: {history['losses'][-1]:.5f}")
    theta = twopl.ability.detach().cpu().numpy()
    a     = twopl.discrimination.detach().cpu().numpy()
    b     = twopl.difficulty.detach().cpu().numpy()
    a_full = np.full(n_items, float(a.mean())); a_full[dense_indices] = a
    b_full = np.full(n_items, float(b.mean())); b_full[dense_indices] = b

    # ── 5. Subject lookup ──────────────────────────────────────────────────────
    subject_id_lookup, subject_name_lookup = {}, {}
    for sid in subject_ids:
        i       = subject_index[sid]
        ability = float(theta[i])
        subject_id_lookup[sid] = ability
        display_name = subject_texts[i].split("\n")[0].replace("Name: ", "").strip()
        subject_name_lookup[display_name] = ability
    mean_theta = float(theta.mean())
    subject_id_lookup["__mean__"]   = mean_theta
    subject_name_lookup["__mean__"] = mean_theta

    ranked = sorted([(k, v) for k, v in subject_name_lookup.items() if k != "__mean__"],
                    key=lambda x: x[1], reverse=True)
    print("\nTop 5:"); [print(f"  {n[:50]:50s}  {v:.3f}") for n, v in ranked[:5]]
    print("Bottom 5:"); [print(f"  {n[:50]:50s}  {v:.3f}") for n, v in ranked[-5:]]

    # ── 6. Encode with all-mpnet-base-v2 ───────────────────────────────────────
    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    print(f"\nLoaded encoder: {ENCODER_NAME}")
    print(f"Encoder embedding dim: {encoder.get_sentence_embedding_dimension()}")
    assert encoder.get_sentence_embedding_dimension() == ENCODER_DIM, (
        f"Encoder dim mismatch: got {encoder.get_sentence_embedding_dimension()}, "
        f"expected {ENCODER_DIM}"
    )

    print(f"\nEncoding {len(item_texts_dense):,} dense items...")
    V_dense = encoder.encode(item_texts_dense, batch_size=128,
                             show_progress_bar=True, convert_to_numpy=True)
    V_all = np.zeros((n_items, ENCODER_DIM), dtype=np.float32)
    V_all[dense_indices] = V_dense.astype(np.float32)
    print(f"Item embeddings: {V_all.shape}")

    print(f"\nEncoding {n_subjects:,} subjects...")
    V_subjects = encoder.encode(subject_texts, batch_size=128,
                                show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    print(f"Subject embeddings: {V_subjects.shape}")

    # Save subject embeddings immediately (model.py needs this artifact).
    np.save(f"{ARTIFACTS_DIR}/subject_embeddings.npy", V_subjects)

    # subject_emb_index keyed by display_name (what the test-time lookup uses)
    subject_emb_index = {}
    for sid in subject_ids:
        i = subject_index[sid]
        display_name = subject_texts[i].split("\n")[0].replace("Name: ", "").strip()
        subject_emb_index[display_name] = int(i)

    # ── 7. Encode unique benchmarks & conditions as text ───────────────────────
    unique_benchmarks = sorted(set(dense_benchmarks))
    unique_conditions = sorted(set(dense_conditions))
    print(f"\nEncoding {len(unique_benchmarks)} unique benchmarks "
          f"and {len(unique_conditions)} unique conditions...")

    bm_texts   = [f"Benchmark: {b}" for b in unique_benchmarks]
    cond_texts = [f"Condition: {c}" for c in unique_conditions]
    bm_emb_matrix   = encoder.encode(bm_texts,   batch_size=64,
                                     convert_to_numpy=True).astype(np.float32)
    cond_emb_matrix = encoder.encode(cond_texts, batch_size=64,
                                     convert_to_numpy=True).astype(np.float32)
    bm_idx   = {b: i for i, b in enumerate(unique_benchmarks)}
    cond_idx = {c: i for i, c in enumerate(unique_conditions)}
    bm_emb_lookup   = {b: bm_emb_matrix[bm_idx[b]].tolist()   for b in unique_benchmarks}
    cond_emb_lookup = {c: cond_emb_matrix[cond_idx[c]].tolist() for c in unique_conditions}

    # ── 8. Cold-start benchmark split & mean-acc maps ──────────────────────────
    bm_counts = pd.Series(dense_benchmarks).value_counts().sort_values()
    candidates = [b for b in bm_counts.index if bm_counts[b] >= 50]
    rng_split = np.random.default_rng(42)
    if len(candidates) >= COLD_START_HOLDOUT_BENCHMARKS:
        heldout_benchmarks = sorted(
            rng_split.choice(candidates, size=COLD_START_HOLDOUT_BENCHMARKS,
                             replace=False).tolist()
        )
    else:
        heldout_benchmarks = sorted(candidates)
    print(f"\nCold-start holdout benchmarks: {heldout_benchmarks}")

    # mean-acc maps computed on NON-heldout rows only (simulating cold-start: at
    # test time, unseen benchmarks fall back to the global mean)
    train_only_df = train_df[~train_df["benchmark"].isin(heldout_benchmarks)]
    global_mean_acc = float(train_only_df["label"].mean())

    subject_mean_acc_by_sid = (
        train_only_df.groupby("subject_id")["label"].mean().to_dict()
    )
    subject_mean_acc_lookup = {}
    for sid, acc in subject_mean_acc_by_sid.items():
        i = subject_index[sid]
        display_name = subject_texts[i].split("\n")[0].replace("Name: ", "").strip()
        subject_mean_acc_lookup[display_name] = float(acc)

    benchmark_mean_acc_lookup = {
        bm: float(acc) for bm, acc in
        train_only_df.groupby("benchmark")["label"].mean().to_dict().items()
    }
    print(f"global_mean_acc (non-heldout): {global_mean_acc:.4f}")

    # ── 9. Build per-split feature matrices (memory-conscious) ─────────────────
    # Avoid materializing one giant X_all (3.4M × 3075 ≈ 42 GB) followed by
    # slice copies and a normalized copy. Instead, compute the index arrays for
    # each split first, then build only that split's X matrix and normalize
    # in-place. Peak memory ≈ size of the train split.
    rows_i, rows_j = np.where(~np.isnan(R_dense))
    y_all          = R_dense[rows_i, rows_j].astype(np.float32)
    orig_j         = dense_indices[rows_j]
    bm_row_idx     = np.array([bm_idx[dense_benchmarks[j]]   for j in rows_j], dtype=np.int64)
    cond_row_idx   = np.array([cond_idx[dense_conditions[j]] for j in rows_j], dtype=np.int64)

    subject_mean_acc_per_subject = np.full(n_subjects, global_mean_acc, dtype=np.float32)
    for sid, i in subject_index.items():
        if sid in subject_mean_acc_by_sid:
            subject_mean_acc_per_subject[i] = float(subject_mean_acc_by_sid[sid])

    benchmark_mean_acc_per_denseitem = np.full(len(dense_indices), global_mean_acc,
                                               dtype=np.float32)
    for k, bm in enumerate(dense_benchmarks):
        if bm in benchmark_mean_acc_lookup:
            benchmark_mean_acc_per_denseitem[k] = benchmark_mean_acc_lookup[bm]

    # Split indices: cold-start vs (Platt holdout vs train).
    row_benchmarks = np.array([dense_benchmarks[j] for j in rows_j])
    cs_mask        = np.isin(row_benchmarks, heldout_benchmarks)
    cs_idx         = np.where(cs_mask)[0]
    keep_idx       = np.where(~cs_mask)[0]
    rng_platt      = np.random.default_rng(7)
    perm           = rng_platt.permutation(len(keep_idx))
    n_platt        = max(1000, int(PLATT_HOLDOUT_FRAC * len(keep_idx)))
    platt_idx      = keep_idx[perm[:n_platt]]
    train_idx      = keep_idx[perm[n_platt:]]
    print(f"\nCold-start rows: {len(cs_idx):,}   "
          f"Platt rows: {len(platt_idx):,}   MLP train rows: {len(train_idx):,}")

    expected_input_dim = 1 + 4 * ENCODER_DIM + 2

    def build_X(idx: np.ndarray) -> np.ndarray:
        si = rows_i[idx]; oj = orig_j[idx]
        bri = bm_row_idx[idx]; cri = cond_row_idx[idx]
        rj  = rows_j[idx]
        return np.hstack([
            theta[si].reshape(-1, 1).astype(np.float32, copy=False),
            V_subjects[si],
            V_all[oj],
            bm_emb_matrix[bri],
            cond_emb_matrix[cri],
            subject_mean_acc_per_subject[si].reshape(-1, 1),
            benchmark_mean_acc_per_denseitem[rj].reshape(-1, 1),
        ])

    print("Building X_train ...")
    X_train = build_X(train_idx); y_train = y_all[train_idx]
    print(f"  X_train: {X_train.shape}")
    print("Building X_platt ...")
    X_platt = build_X(platt_idx); y_platt = y_all[platt_idx]
    print(f"  X_platt: {X_platt.shape}")
    print("Building X_cs ...")
    X_cs    = build_X(cs_idx);    y_cs    = y_all[cs_idx]
    print(f"  X_cs:    {X_cs.shape}")

    input_dim = X_train.shape[1]
    print(f"input_dim={input_dim}   expected={expected_input_dim}")
    assert input_dim == expected_input_dim, (
        f"input_dim mismatch: got {input_dim}, expected {expected_input_dim}"
    )

    # ── 10. Normalize in-place ─────────────────────────────────────────────────
    X_mean = X_train.mean(axis=0).astype(np.float32)
    X_std  = (X_train.std(axis=0) + 1e-8).astype(np.float32)
    for arr in (X_train, X_platt, X_cs):
        arr -= X_mean
        arr /= X_std
    print(f"X_train (normalized) mean: {X_train.mean():.4f}   std: {X_train.std():.4f}")

    # Free arrays no longer needed.
    del rows_i, rows_j, orig_j, bm_row_idx, cond_row_idx, row_benchmarks
    del cs_mask, cs_idx, keep_idx, perm, platt_idx, train_idx
    del V_all, V_subjects, bm_emb_matrix, cond_emb_matrix
    del subject_mean_acc_per_subject, benchmark_mean_acc_per_denseitem
    gc.collect()

    # ── 11. Fit MLP ────────────────────────────────────────────────────────────
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

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)

    mlp       = ResponseMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining MLP...")
    N_EPOCHS = 15
    for epoch in range(N_EPOCHS):
        mlp.train()
        total_loss, n_batches = 0.0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device); y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = mlp(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()
        print(f"  Epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/n_batches:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    mlp.eval()
    # Free training tensor & numpy now that MLP is fit.
    del dataset, loader
    gc.collect()

    def _forward_logits(x_n_np: np.ndarray, batch: int = 8192) -> np.ndarray:
        out = []
        with torch.no_grad():
            for k in range(0, len(x_n_np), batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(x_n_np[k:k+batch])
                ).to(device)
                out.append(mlp(xb).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)

    # Sanity check on first 1000 train rows.
    train_logits = _forward_logits(X_train[:1000])
    train_probs  = 1.0 / (1.0 + np.exp(-train_logits))
    print(f"\nSanity (first 1000 train rows): "
          f"pred mean={train_probs.mean():.3f}  "
          f"range=[{train_probs.min():.3f}, {train_probs.max():.3f}]")
    del X_train, y_train
    gc.collect()

    # ── 12. Platt calibration ──────────────────────────────────────────────────
    print("\nFitting Platt calibration on holdout logits...")
    platt_logits = _forward_logits(X_platt)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(platt_logits.reshape(-1, 1), y_platt.astype(np.int64))
    platt_a = float(lr.coef_[0, 0])
    platt_b = float(lr.intercept_[0])
    print(f"Platt: a={platt_a:.4f}  b={platt_b:.4f}")
    del X_platt
    gc.collect()

    def _calibrated_prob(logits: np.ndarray) -> np.ndarray:
        z = platt_a * logits + platt_b
        return 1.0 / (1.0 + np.exp(-z))

    def _nll(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
        p = np.clip(p, eps, 1.0 - eps)
        return float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # ── 13. Cold-start evaluation ──────────────────────────────────────────────
    if len(y_cs) > 0:
        cs_logits = _forward_logits(X_cs)
        cs_probs_raw = 1.0 / (1.0 + np.exp(-cs_logits))
        cs_probs_cal = _calibrated_prob(cs_logits)
        cs_nll_raw = _nll(y_cs, cs_probs_raw)
        cs_nll_cal = _nll(y_cs, cs_probs_cal)
        platt_nll  = _nll(y_platt, _calibrated_prob(platt_logits))
        print(f"\nPlatt holdout NLL (calibrated): {platt_nll:.4f}  (higher is better)")
        print(f"COLD-START NLL  (raw):         {cs_nll_raw:.4f}")
        print(f"COLD-START NLL  (calibrated):  {cs_nll_cal:.4f}")
        print(f"Held-out benchmarks: {heldout_benchmarks}")
    else:
        print("\nNo cold-start rows — skipping cold-start NLL")
        cs_nll_raw = cs_nll_cal = None
    del X_cs, y_cs, y_platt
    gc.collect()

    # ── 14. Fit k-means centroids (for diversity in labeling.py) ───────────────
    print("\nFitting k-means centroids on item embeddings...")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(item_texts_dense),
                            size=min(50000, len(item_texts_dense)), replace=False)
    X_cent = V_dense[sample_idx]
    km = MiniBatchKMeans(n_clusters=64, n_init=10, random_state=42, batch_size=4096)
    km.fit(X_cent)
    print(f"Centroids: {km.cluster_centers_.shape}")

    # ── 15. Save artifacts ─────────────────────────────────────────────────────
    # subject_embeddings.npy was already saved right after encoding (before the
    # per-split memory cleanup). item_embeddings.npy is NOT needed at inference
    # (model.py encodes items fresh) and we've already freed V_all to reduce
    # memory pressure during training.
    A = ARTIFACTS_DIR
    np.save(f"{A}/theta.npy",              theta)
    np.save(f"{A}/item_a.npy",             a_full)
    np.save(f"{A}/item_b.npy",             b_full)
    np.save(f"{A}/centroids.npy",          km.cluster_centers_.astype(np.float32))
    np.save(f"{A}/X_mean.npy",             X_mean.astype(np.float32))
    np.save(f"{A}/X_std.npy",              X_std.astype(np.float32))

    json.dump(subject_id_lookup,       open(f"{A}/subject_id_lookup.json",      "w"))
    json.dump(subject_name_lookup,     open(f"{A}/subject_name_lookup.json",    "w"))
    json.dump(subject_emb_index,       open(f"{A}/subject_emb_index.json",      "w"))
    json.dump(bm_emb_lookup,           open(f"{A}/bm_emb_lookup.json",          "w"))
    json.dump(cond_emb_lookup,         open(f"{A}/cond_emb_lookup.json",        "w"))
    json.dump(subject_mean_acc_lookup, open(f"{A}/subject_mean_acc.json",       "w"))
    json.dump(benchmark_mean_acc_lookup, open(f"{A}/benchmark_mean_acc.json",   "w"))
    json.dump({"global_mean_acc": global_mean_acc},
              open(f"{A}/global_mean_acc.json", "w"))
    json.dump({"a": platt_a, "b": platt_b}, open(f"{A}/platt.json", "w"))
    json.dump({"input_dim": input_dim}, open(f"{A}/mlp_config.json",            "w"))

    torch.save(mlp.state_dict(), f"{A}/mlp.pt")

    bundle = {
        "encoder_name":      ENCODER_NAME,
        "encoder_dim":       ENCODER_DIM,
        "input_dim":         input_dim,
        "heldout_benchmarks": heldout_benchmarks,
        "cold_start_nll_raw":        cs_nll_raw,
        "cold_start_nll_calibrated": cs_nll_cal,
    }
    pickle.dump(bundle, open(f"{A}/bundle.pkl", "wb"))

    volume.commit()
    print(f"\nAll artifacts saved")
    print(f"input_dim = {input_dim}  (expected {expected_input_dim})")
    return "Training complete"


@app.local_entrypoint()
def main():
    print("Starting training on Modal GPU...")
    result = run_training.remote()
    print(result)

    print("\nDownloading artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    vol = modal.Volume.from_name("eval-artifacts")
    for fname in [
        "theta.npy", "item_a.npy", "item_b.npy",
        "centroids.npy", "item_embeddings.npy", "subject_embeddings.npy",
        "X_mean.npy", "X_std.npy", "mlp.pt",
        "subject_id_lookup.json", "subject_name_lookup.json",
        "subject_emb_index.json",
        "bm_emb_lookup.json", "cond_emb_lookup.json",
        "subject_mean_acc.json", "benchmark_mean_acc.json",
        "global_mean_acc.json", "platt.json",
        "mlp_config.json", "bundle.pkl",
    ]:
        data = b"".join(vol.read_file(fname))
        with open(f"artifacts/{fname}", "wb") as f:
            f.write(data)
        print(f"  Downloaded {fname}")

    print("\nDone. Run: python validate.py")
