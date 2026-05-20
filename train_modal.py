"""
train_modal.py — Run the full training pipeline on Modal GPU.

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


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/artifacts": volume},
)
def run_training():
    import sys
    sys.path.insert(0, "/tmp/torch_measure/src")

    import os, json, pickle
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
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

    # ── 6. Encode dense items ──────────────────────────────────────────────────
    ENCODER_NAME = "paraphrase-MiniLM-L3-v2"
    encoder      = SentenceTransformer(ENCODER_NAME)
    print(f"\nEncoding {len(item_texts_dense):,} dense items...")
    V_dense = encoder.encode(item_texts_dense, batch_size=256,
                             show_progress_bar=True, convert_to_numpy=True)
    V_all = np.zeros((n_items, V_dense.shape[1]), dtype=np.float32)
    V_all[dense_indices] = V_dense
    print(f"Embeddings: {V_all.shape}")

    # ── 7. Build feature matrix ────────────────────────────────────────────────
    bm_dummies   = pd.get_dummies(pd.Series(dense_benchmarks), prefix="bm").astype(np.float32)
    cond_dummies = pd.get_dummies(pd.Series(dense_conditions), prefix="cond").astype(np.float32)
    bm_columns   = bm_dummies.columns.tolist()
    cond_columns = cond_dummies.columns.tolist()

    rows_i, rows_j = np.where(~np.isnan(R_dense))
    y_train        = R_dense[rows_i, rows_j].astype(np.float32)
    orig_j         = dense_indices[rows_j]
    X_train        = np.hstack([
        theta[rows_i].reshape(-1, 1),
        V_all[orig_j],
        bm_dummies.values[rows_j],
        cond_dummies.values[rows_j],
    ]).astype(np.float32)
    input_dim = X_train.shape[1]
    print(f"\nTraining pairs: {len(y_train):,}  Input dim: {input_dim}")
    print(f"Label mean: {y_train.mean():.3f}")

    # ── 8. Normalize features ──────────────────────────────────────────────────
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    X_norm = (X_train - X_mean) / X_std

    # Sanity check normalization
    print(f"X_norm mean: {X_norm.mean():.4f}  std: {X_norm.std():.4f}")

    # ── 9. Fit MLP ─────────────────────────────────────────────────────────────
    class ResponseMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),        nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    dataset = TensorDataset(
        torch.tensor(X_norm,  dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=2)

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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = mlp(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.5f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # Sanity check predictions
    mlp.eval()
    with torch.no_grad():
        sample = torch.tensor(X_norm[:1000], dtype=torch.float32).to(device)
        logits = mlp(sample).cpu().numpy()
        probs  = 1 / (1 + np.exp(-logits))
    print(f"\nSanity check on 1000 train samples:")
    print(f"  Pred mean: {probs.mean():.3f}  std: {probs.std():.3f}")
    print(f"  Pred range: [{probs.min():.3f}, {probs.max():.3f}]")

    # ── 10. Fit k-means centroids ──────────────────────────────────────────────
    print("\nFitting k-means centroids...")
    rng        = np.random.default_rng(0)
    sample_idx = rng.choice(len(item_texts_dense),
                            size=min(50000, len(item_texts_dense)), replace=False)
    X_cent = encoder.encode([item_texts_dense[j] for j in sample_idx],
                            batch_size=256, show_progress_bar=True,
                            convert_to_numpy=True)
    km = MiniBatchKMeans(n_clusters=64, n_init=10, random_state=42, batch_size=4096)
    km.fit(X_cent)
    print(f"Centroids: {km.cluster_centers_.shape}")

    # ── 11. Save artifacts ─────────────────────────────────────────────────────
    A = ARTIFACTS_DIR
    np.save(f"{A}/theta.npy",           theta)
    np.save(f"{A}/item_a.npy",          a_full)
    np.save(f"{A}/item_b.npy",          b_full)
    np.save(f"{A}/centroids.npy",       km.cluster_centers_)
    np.save(f"{A}/item_embeddings.npy", V_all)
    np.save(f"{A}/X_mean.npy",          X_mean)
    np.save(f"{A}/X_std.npy",           X_std)

    json.dump(subject_id_lookup,   open(f"{A}/subject_id_lookup.json",   "w"))
    json.dump(subject_name_lookup, open(f"{A}/subject_name_lookup.json", "w"))
    json.dump(bm_columns,          open(f"{A}/bm_columns.json",          "w"))
    json.dump(cond_columns,        open(f"{A}/cond_columns.json",        "w"))
    json.dump({"input_dim": input_dim}, open(f"{A}/mlp_config.json",     "w"))

    torch.save(mlp.state_dict(), f"{A}/mlp.pt")

    bundle = {
        "bm_columns":       bm_columns,
        "cond_columns":     cond_columns,
        "encoder_name":     ENCODER_NAME,
        "n_embedding_dims": V_dense.shape[1],
        "input_dim":        input_dim,
    }
    pickle.dump(bundle, open(f"{A}/bundle.pkl", "wb"))

    volume.commit()
    print(f"\nAll artifacts saved  done")
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
        "centroids.npy", "item_embeddings.npy",
        "X_mean.npy", "X_std.npy", "mlp.pt",
        "subject_id_lookup.json", "subject_name_lookup.json",
        "bm_columns.json", "cond_columns.json",
        "mlp_config.json", "bundle.pkl",
    ]:
        data = b"".join(vol.read_file(fname))
        with open(f"artifacts/{fname}", "wb") as f:
            f.write(data)
        print(f"  Downloaded {fname}")

    print("\nDone. Run: python validate.py")