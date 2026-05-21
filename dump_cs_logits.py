"""
Standalone Modal job: load the trained MLP + artifacts from the eval-artifacts
volume, reconstruct the cold-start hold-out features (bfcl + matharena),
forward-pass, save (cs_logits.npy, cs_y.npy) back to the volume.

We then fit T locally to minimize NLL on cold-start.

Does NOT retrain anything. Steps are a strict subset of train_modal.py
(sections 1-3, 6-9 of that file) without IRT/MLP/Platt/k-means fitting.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "numpy>=1.26", "pandas>=2.0",
        "scikit-learn>=1.3", "sentence-transformers>=2.7",
        "datasets>=2.14", "huggingface_hub",
    )
    .apt_install("git")
)

app    = modal.App("dump-cs-logits", image=image)
volume = modal.Volume.from_name("eval-artifacts", create_if_missing=False)

ENCODER_NAME = "all-mpnet-base-v2"
ENCODER_DIM  = 768
COLD_START_HOLDOUT_BENCHMARKS = 2


@app.function(
    image=image,
    gpu="T4",
    timeout=2400,
    memory=32768,
    cpu=4.0,
    volumes={"/artifacts": volume},
)
def run_dump():
    import os, json, pickle
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    REPO_ID = "aims-foundations/measurement-db"
    REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    A = "/artifacts"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load data (same as train_modal.py §1) ──────────────────────────────
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
    print(f"Responses: {len(responses):,}  Items: {len(items):,}")

    # ── 2. Build train_df + binarize (same as train_modal.py §2) ──────────────
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

    df["label"] = df.apply(binarize, axis=1)
    train_df = df[["subject_id", "item_id", "benchmark", "condition",
                   "subject_content", "item_content", "label"]].copy()
    print(f"train_df: {len(train_df):,} rows")

    # ── 3. Response matrix (same as train_modal.py §3) ────────────────────────
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
    print(f"Dense items: {len(dense_indices):,}")

    item_id_ordered    = sorted(item_index, key=lambda x: item_index[x])
    item_text_map      = train_df.drop_duplicates("item_id").set_index("item_id")["item_content"]
    item_meta_map      = (train_df.drop_duplicates("item_id")
                          .set_index("item_id")[["benchmark", "condition"]].astype(str))
    item_texts         = [str(item_text_map[iid])    for iid in item_id_ordered]
    dense_item_ids     = [item_id_ordered[j]         for j   in dense_indices]
    item_texts_dense   = [item_texts[j]              for j   in dense_indices]
    dense_benchmarks   = [item_meta_map.loc[iid, "benchmark"] if iid in item_meta_map.index
                          else "unknown" for iid in dense_item_ids]
    dense_conditions   = [item_meta_map.loc[iid, "condition"] if iid in item_meta_map.index
                          else "none"    for iid in dense_item_ids]

    # ── 4. Load saved artifacts (theta, subject_emb, X_mean/std, MLP) ─────────
    theta              = np.load(f"{A}/theta.npy")
    V_subjects         = np.load(f"{A}/subject_embeddings.npy")
    X_mean             = np.load(f"{A}/X_mean.npy")
    X_std              = np.load(f"{A}/X_std.npy")
    mlp_config         = json.load(open(f"{A}/mlp_config.json"))
    input_dim          = int(mlp_config["input_dim"])
    print(f"Loaded artifacts. input_dim={input_dim}")

    # subject_mean_acc + benchmark_mean_acc + global_mean lookups
    subj_mean_lookup   = json.load(open(f"{A}/subject_mean_acc.json"))
    bm_mean_lookup     = json.load(open(f"{A}/benchmark_mean_acc.json"))
    global_mean_acc    = json.load(open(f"{A}/global_mean_acc.json"))["global_mean_acc"]
    # Read heldout benchmark list from bundle (use the exact same split as training)
    bundle             = pickle.load(open(f"{A}/bundle.pkl", "rb"))
    heldout_benchmarks = bundle["heldout_benchmarks"]
    print(f"Held-out benchmarks: {heldout_benchmarks}")

    subject_texts_for_name = {}
    for sid in subject_ids:
        i = subject_index[sid]
        sub_content = (
            train_df[train_df.subject_id == sid].iloc[0].subject_content
            if (train_df.subject_id == sid).any() else ""
        )
        display_name = sub_content.split("\n")[0].replace("Name: ", "").strip()
        subject_texts_for_name[i] = display_name

    # Build per-subject mean-acc vector aligned with subject_index
    subject_mean_acc_per_subject = np.full(n_subjects, global_mean_acc, dtype=np.float32)
    for sid in subject_ids:
        i = subject_index[sid]
        name = subject_texts_for_name[i]
        if name in subj_mean_lookup:
            subject_mean_acc_per_subject[i] = float(subj_mean_lookup[name])

    benchmark_mean_acc_per_denseitem = np.full(len(dense_indices), global_mean_acc,
                                                dtype=np.float32)
    for k, bm in enumerate(dense_benchmarks):
        if bm in bm_mean_lookup:
            benchmark_mean_acc_per_denseitem[k] = bm_mean_lookup[bm]

    # ── 5. Build cold-start row/index structure FIRST, then encode only
    #       the subset of items we need (massive speedup vs encoding all 38k).
    rows_i_full, rows_j_full = np.where(~np.isnan(R_dense))
    y_all_full     = R_dense[rows_i_full, rows_j_full].astype(np.float32)
    row_benchmarks = np.array([dense_benchmarks[j] for j in rows_j_full])
    cs_mask        = np.isin(row_benchmarks, heldout_benchmarks)
    cs_idx         = np.where(cs_mask)[0]
    print(f"Cold-start rows: {len(cs_idx):,}")

    # Which dense items appear in the cold-start subset?
    cs_dense_items = np.unique(rows_j_full[cs_idx])
    print(f"Distinct cold-start dense items: {len(cs_dense_items):,}")

    cs_item_texts = [item_texts_dense[j] for j in cs_dense_items]
    cs_item_pos   = {int(j): k for k, j in enumerate(cs_dense_items)}

    # ── 6. Encode the relevant items + benchmarks + conditions ────────────────
    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    print(f"\nEncoding {len(cs_item_texts):,} cold-start items "
          f"(vs {len(item_texts_dense):,} total dense)...")
    V_cs_items = encoder.encode(cs_item_texts, batch_size=128,
                                show_progress_bar=True, convert_to_numpy=True
                                ).astype(np.float32)

    unique_benchmarks = sorted(set(dense_benchmarks))
    unique_conditions = sorted(set(dense_conditions))
    bm_texts   = [f"Benchmark: {b}" for b in unique_benchmarks]
    cond_texts = [f"Condition: {c}" for c in unique_conditions]
    bm_emb_matrix   = encoder.encode(bm_texts,   batch_size=64,
                                     convert_to_numpy=True).astype(np.float32)
    cond_emb_matrix = encoder.encode(cond_texts, batch_size=64,
                                     convert_to_numpy=True).astype(np.float32)
    bm_idx   = {b: i for i, b in enumerate(unique_benchmarks)}
    cond_idx = {c: i for i, c in enumerate(unique_conditions)}

    # ── 7. Restrict everything to cs_idx rows and build X_cs ──────────────────
    rows_i  = rows_i_full[cs_idx]
    rows_j  = rows_j_full[cs_idx]
    y_cs    = y_all_full[cs_idx]
    bm_row_idx   = np.array([bm_idx[dense_benchmarks[j]]   for j in rows_j], dtype=np.int64)
    cond_row_idx = np.array([cond_idx[dense_conditions[j]] for j in rows_j], dtype=np.int64)
    cs_item_row_idx = np.array([cs_item_pos[int(j)] for j in rows_j], dtype=np.int64)

    print("Building X_cs ...")
    X_cs = np.hstack([
        theta[rows_i].reshape(-1, 1).astype(np.float32, copy=False),
        V_subjects[rows_i],
        V_cs_items[cs_item_row_idx],
        bm_emb_matrix[bm_row_idx],
        cond_emb_matrix[cond_row_idx],
        subject_mean_acc_per_subject[rows_i].reshape(-1, 1),
        benchmark_mean_acc_per_denseitem[rows_j].reshape(-1, 1),
    ])
    print(f"  X_cs: {X_cs.shape}")
    X_cs -= X_mean; X_cs /= X_std

    # ── 7. Load MLP and forward pass ──────────────────────────────────────────
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
        def forward(self, x): return self.net(x).squeeze(-1)

    mlp = ResponseMLP(input_dim).to(device)
    mlp.load_state_dict(torch.load(f"{A}/mlp.pt", map_location=device))
    mlp.eval()

    out = []
    with torch.no_grad():
        for k in range(0, len(X_cs), 8192):
            xb = torch.from_numpy(np.ascontiguousarray(X_cs[k:k+8192])).to(device)
            out.append(mlp(xb).cpu().numpy())
    cs_logits = np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)

    # Sanity: NLL on raw probs (should be close to bundle.cold_start_nll_raw)
    p_raw = 1.0 / (1.0 + np.exp(-cs_logits))
    p_raw = np.clip(p_raw, 1e-7, 1 - 1e-7)
    nll_raw = float(np.mean(y_cs * np.log(p_raw) + (1 - y_cs) * np.log(1 - p_raw)))
    print(f"\nCold-start logits: {cs_logits.shape}, range [{cs_logits.min():.2f}, {cs_logits.max():.2f}]")
    print(f"NLL_raw recomputed: {nll_raw:.4f}  (bundle reports {bundle['cold_start_nll_raw']:.4f})")

    np.save(f"{A}/cs_logits.npy", cs_logits.astype(np.float32))
    np.save(f"{A}/cs_y.npy",      y_cs.astype(np.float32))

    # ── 8. Fit T locally too (cheap, scalar optimization) ────────────────────
    from scipy.optimize import minimize_scalar

    def nll_at_T(T):
        T = max(T, 0.01)
        p = 1.0 / (1.0 + np.exp(-cs_logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.mean(y_cs * np.log(p) + (1 - y_cs) * np.log(1 - p)))

    res = minimize_scalar(nll_at_T, bounds=(0.1, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    T_star = float(res.x)
    nll_star = -res.fun
    print(f"\nOptimal T on cold-start val: T*={T_star:.4f}")
    print(f"NLL(T=1)      = {nll_raw:.4f}")
    print(f"NLL(T*={T_star:.3f}) = {nll_star:.4f}")
    print(f"Δ NLL         = {nll_star - nll_raw:+.4f}")

    json.dump({"T": T_star, "nll_T1": nll_raw, "nll_Tstar": nll_star,
               "fit_set": "cold_start_holdout",
               "heldout_benchmarks": heldout_benchmarks},
              open(f"{A}/temperature.json", "w"))

    volume.commit()
    print("\nSaved: cs_logits.npy, cs_y.npy, temperature.json")
    return T_star


@app.local_entrypoint()
def main():
    T_star = run_dump.remote()
    print(f"\nLocal: optimal T = {T_star:.4f}")
