"""Build a Codabench submission ZIP from a manifest and record it in the ledger.

The submission flow is intentionally manifest-driven and ZIP-deterministic:

1. A JSON manifest under ``manifests/`` declares which files belong to a
   submission, the model name, hyperparameters, optional offline validation
   numbers, and free-form notes.
2. ``submit.py build manifests/<name>.json`` materializes a ZIP at
   ``runs/zips/<ts>__<model_name>__<short_sha>.zip``, deterministically (sorted
   entries, fixed mtime) so the ZIP's sha256 is stable across machines.
3. The same call inserts a row in ``runs/ledger.db`` BEFORE upload, capturing
   the manifest sha + git context. Upload to Codabench is manual until/unless
   we get an API.
4. Once the round resolves, ``submit.py update <id> --leaderboard-nll X
   --leaderboard-auc Y --round-id Z`` patches the ledger row.

The point: the ZIP we upload to Codabench and the one referenced by
``manifest_sha`` in the ledger are bit-for-bit identical; the `model.py` at
the repo root (used for grading per the assignment instructions) is the
exact source for the best-scoring submission's ZIP.

Usage::

    python submit.py build manifests/s1_constant.json
    python submit.py list
    python submit.py update 7 --leaderboard-nll -0.612 --round-id 2026-05-02-r1
    python submit.py show 7
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import ledger

REPO_ROOT = Path(__file__).resolve().parent
ZIP_DIR = REPO_ROOT / "runs" / "zips"

# Fixed timestamp baked into every ZipInfo so the resulting bytes are
# deterministic (zipfile defaults to time.localtime()).
DETERMINISTIC_MTIME = (1980, 1, 1, 0, 0, 0)


@dataclass
class Manifest:
    """A normalized view of a submission manifest JSON."""

    model_name: str
    files: list[Path]
    notes: str | None
    hyperparams: dict[str, Any]
    val_nll_mean: float | None
    val_nll_std: float | None
    val_auc: float | None

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raw = json.loads(path.read_text())
        if "model_name" not in raw or "files" not in raw:
            raise ValueError(
                f"Manifest {path} must declare at least 'model_name' and 'files'"
            )
        files = [(REPO_ROOT / f).resolve() for f in raw["files"]]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"Manifest {path} references missing file: {f}")
            try:
                f.relative_to(REPO_ROOT)
            except ValueError as e:
                raise ValueError(
                    f"Manifest {path} references file outside repo: {f}"
                ) from e
        return cls(
            model_name=str(raw["model_name"]),
            files=files,
            notes=raw.get("notes"),
            hyperparams=dict(raw.get("hyperparams", {})),
            val_nll_mean=_opt_float(raw.get("val_nll_mean")),
            val_nll_std=_opt_float(raw.get("val_nll_std")),
            val_auc=_opt_float(raw.get("val_auc")),
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.decode().strip() or None


def build_zip(manifest: Manifest, dest: Path) -> str:
    """Write a deterministic ZIP for the manifest. Returns its sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with zipfile.ZipFile(dest, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(manifest.files, key=lambda p: str(p)):
            arcname = str(f.relative_to(REPO_ROOT))
            data = f.read_bytes()
            info = zipfile.ZipInfo(filename=arcname, date_time=DETERMINISTIC_MTIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
            sha.update(arcname.encode())
            sha.update(b"\0")
            sha.update(data)
    return sha.hexdigest()


def _short(sha: str | None, n: int = 8) -> str:
    return (sha or "nogit")[:n]


# -- subcommands ------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = Manifest.load(manifest_path)

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit_sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    zip_name = f"{ts}__{manifest.model_name}__{_short(commit_sha)}{'-dirty' if dirty else ''}.zip"

    # Codabench enforces a 64-character cap on uploaded ZIP filenames.
    # We refuse to silently truncate because the manifest's `model_name`
    # is also the ledger key; mismatches between intended and actual name
    # would be confusing later.
    MAX_ZIP_NAME = 64
    if len(zip_name) > MAX_ZIP_NAME:
        overflow = len(zip_name) - MAX_ZIP_NAME
        budget = len(manifest.model_name) - overflow
        raise ValueError(
            f"Generated zip name '{zip_name}' is {len(zip_name)} chars; "
            f"Codabench's limit is {MAX_ZIP_NAME}. "
            f"Shorten manifest.model_name from {len(manifest.model_name)} "
            f"to {budget} chars or fewer."
        )

    zip_path = ZIP_DIR / zip_name

    manifest_sha = build_zip(manifest, zip_path)

    notes = manifest.notes or ""
    if dirty:
        notes = (notes + "\n[warning] built from a dirty working tree").strip()

    sub_id = ledger.record_submission(
        ts=ts,
        manifest_sha=manifest_sha,
        zip_path=str(zip_path.relative_to(REPO_ROOT)),
        model_name=manifest.model_name,
        branch=branch,
        commit_sha=commit_sha,
        hyperparams=manifest.hyperparams,
        val_nll_mean=manifest.val_nll_mean,
        val_nll_std=manifest.val_nll_std,
        val_auc=manifest.val_auc,
        notes=notes or None,
    )

    print(f"id          : {sub_id}")
    print(f"model       : {manifest.model_name}")
    print(f"zip         : {zip_path}")
    print(f"manifest_sha: {manifest_sha}")
    print(f"branch      : {branch}  ({_short(commit_sha)}{' dirty' if dirty else ''})")
    print()
    print("Upload manually at: https://aimslab.stanford.edu/competition/submit")
    print(
        "After the round resolves, patch the row with:\n"
        f"  python submit.py update {sub_id} --leaderboard-nll <NLL> "
        "--leaderboard-auc <AUC> --round-id <ROUND>"
    )
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    ledger.update_score(
        args.submission_id,
        leaderboard_round_id=args.round_id,
        leaderboard_nll=args.leaderboard_nll,
        leaderboard_auc=args.leaderboard_auc,
        uploaded_at=args.uploaded_at,
        notes_append=args.note,
    )
    print(f"Updated submission {args.submission_id}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = ledger.list_submissions(
        model_name=args.model, only_scored=args.scored_only, limit=args.limit
    )
    if not rows:
        print("(no submissions yet)")
        return 0
    fmt = (
        "{id:>4}  {ts}  {model:<20}  "
        "val_nll={val:>7}  lb_nll={lb:>7}  {sha}  {notes}"
    )
    print(
        fmt.format(
            id="id",
            ts="ts                  ",
            model="model",
            val="val_nll",
            lb="lb_nll ",
            sha="manifest        ",
            notes="notes",
        )
    )
    for r in rows:
        print(
            fmt.format(
                id=r.id,
                ts=r.ts,
                model=r.model_name[:20],
                val=_fmt(r.val_nll_mean),
                lb=_fmt(r.leaderboard_nll),
                sha=r.manifest_sha[:12] + "...",
                notes=(r.notes or "").splitlines()[0] if r.notes else "",
            )
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    rows = ledger.list_submissions(limit=None)
    match = next((r for r in rows if r.id == args.submission_id), None)
    if match is None:
        print(f"No submission with id={args.submission_id}", file=sys.stderr)
        return 1
    print(json.dumps(match.__dict__, indent=2, default=str))
    return 0


def _fmt(x: float | None) -> str:
    return "  -    " if x is None else f"{x:7.4f}"


# -- argparse plumbing ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a submission ZIP from a manifest")
    b.add_argument("manifest", help="Path to manifest JSON")
    b.set_defaults(func=cmd_build)

    u = sub.add_parser("update", help="Patch leaderboard fields on a row")
    u.add_argument("submission_id", type=int)
    u.add_argument("--leaderboard-nll", type=float)
    u.add_argument("--leaderboard-auc", type=float)
    u.add_argument("--round-id", dest="round_id")
    u.add_argument("--uploaded-at")
    u.add_argument("--note", help="Free-form note appended to the notes column")
    u.set_defaults(func=cmd_update)

    l = sub.add_parser("list", help="List submissions, newest first")
    l.add_argument("--model", help="Filter by model name")
    l.add_argument("--scored-only", action="store_true")
    l.add_argument("--limit", type=int, default=20)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="Show a single submission row")
    s.add_argument("submission_id", type=int)
    s.set_defaults(func=cmd_show)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
