#!/usr/bin/env python3
"""Assemble the Gradescope submission bundle from the best ledger row.

Reads `runs/ledger.db`, picks the submission with the highest
`leaderboard_nll` (ties broken by lowest `id`), expands its ZIP into
`gradescope/`, and writes a README.md describing how to reproduce the
artifacts and the expected leaderboard score.

The rubric requires (handbook p.4-5):
  - report.pdf
  - model.py (+ labeling.py if used)
  - models.txt (HuggingFace repos)
  - weight checkpoints (artifacts/)
  - README.md (reproduction steps)
  - the Gradescope bundle must match the best Codabench score
    (if mismatched, Gradescope wins).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LEDGER = REPO_ROOT / "runs" / "ledger.db"
ZIP_DIR = REPO_ROOT / "runs" / "zips"
DEST = REPO_ROOT / "gradescope"
REPORT_PDF = (
    REPO_ROOT.parent / "prediction-evaluation-challenge-overleaf" / "challenge.pdf"
)


def best_submission() -> dict:
    if not LEDGER.exists():
        sys.exit(f"ledger not found: {LEDGER}")
    con = sqlite3.connect(LEDGER)
    con.row_factory = sqlite3.Row
    row = con.execute(
        """
        SELECT id, ts, model_name, leaderboard_round_id,
               leaderboard_nll, leaderboard_auc, zip_path, notes,
               commit_sha, branch
          FROM submissions
         WHERE leaderboard_nll IS NOT NULL
         ORDER BY leaderboard_nll DESC, id ASC
         LIMIT 1
        """
    ).fetchone()
    if row is None:
        sys.exit("no submissions with a recorded leaderboard NLL")
    return dict(row)


def expand(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def write_readme(sub: dict, dest: Path, report_included: bool) -> None:
    readme = dest / "README.md"
    nll = sub["leaderboard_nll"]
    auc = sub["leaderboard_auc"]
    nll_str = f"{nll:.2f}" if isinstance(nll, (int, float)) else str(nll)
    auc_str = f"{auc:.2f}" if isinstance(auc, (int, float)) else str(auc)
    round_id = sub["leaderboard_round_id"] or "(unrecorded)"
    body = f"""# Predictive Evaluation Challenge --- Gradescope bundle

Best Codabench submission: **`{sub['model_name']}`** (ledger id {sub['id']},
round `{round_id}`).

| Metric          | Value     |
| --------------- | --------- |
| Leaderboard NLL | **{nll_str}** |
| Leaderboard AUC | {auc_str}     |

The artifacts in `artifacts/` are byte-identical to those shipped to
Codabench in the matching ZIP (manifest hashes recorded in
`runs/ledger.db`).

## Files

- `model.py` --- `predict()` and `update()` entry points
- `labeling.py` --- `acquisition_function()` for the $K{{=}}5$ adaptive
  labeling channel
- `models.txt` --- HuggingFace encoder repos to pre-fetch
- `requirements.txt` --- Python dependencies (matches the Codabench
  sandbox)
- `artifacts/` --- MLP weights, calibration scalars, IRT lookups, mean
  pass-rates, $k$-means centroids
- `report.pdf` --- 4-page main report + appendix
  ({'included' if report_included else 'NOT FOUND --- attach manually'})

## How to reproduce

1. Install dependencies in a fresh Python 3.11 environment:
   ```
   pip install -r requirements.txt
   ```
2. Fetch the HuggingFace encoder listed in `models.txt`
   (this is the same step Codabench performs in the sandbox setup).
3. Retrain from scratch (optional, end-to-end):
   ```
   modal run --detach train_modal.py::train
   ```
   The full pipeline (IRT fit, mpnet embedding, MLP training, Platt
   fit, temperature fit) runs on a single T4 in ~20-25 minutes from
   the public `aims-foundations/measurement-db` parquet shards. The
   resulting artifacts are deterministic up to PyTorch RNG; expected
   cold-start NLL is `-0.665` at $T{{=}}4.073$.
4. To score in-process (no Modal): call `model.predict(items)` where
   `items` is a list of `{{benchmark, condition, subject_content,
   item_content}}` dicts. `acquisition_function()` selects $K$ items
   to label per round.

## Notes

- Built from commit `{(sub['commit_sha'] or 'unknown')[:8]}` on branch
  `{sub['branch'] or 'unknown'}`. Notes:

  > {(sub['notes'] or '').replace(chr(10), ' ').strip()[:600]}
"""
    readme.write_text(body)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--include-pdf",
        action="store_true",
        help="Copy challenge.pdf from the Overleaf clone into the bundle",
    )
    args = p.parse_args()

    sub = best_submission()
    zip_path = (REPO_ROOT / sub["zip_path"]).resolve()
    if not zip_path.exists():
        sys.exit(f"missing ZIP: {zip_path}")

    expand(zip_path, DEST)
    pdf_included = False
    if args.include_pdf and REPORT_PDF.exists():
        shutil.copy2(REPORT_PDF, DEST / "report.pdf")
        pdf_included = True

    write_readme(sub, DEST, pdf_included)
    print(f"built: {DEST} (from ledger id {sub['id']}, NLL={sub['leaderboard_nll']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
