#!/usr/bin/env python3
r"""Render the LaTeX score-macro block for docs/ablations_section.tex from runs/ledger.db.

Usage:
    python docs/render_ablation_macros.py            # print to stdout
    python docs/render_ablation_macros.py --in-place # rewrite the macro block in
                                                     # docs/ablations_section.tex

The script reads runs/ledger.db, picks the single most-recent row per model_name
(by id), and emits one ``\newcommand`` per LB NLL / AUC value. Missing scores fall
back to ``\TBD``, so it is safe to re-run after every leaderboard round.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "runs" / "ledger.db"
TEX = REPO_ROOT / "docs" / "ablations_section.tex"

MACRO_MAP = {
    # ledger model_name -> (NLL macro, AUC macro)
    "mpnet_platt_meanresid":    ("lbNLLsubOne",       "lbAUCsubOne"),
    "mpnet_noplatt_meanresid":  ("lbNLLsubTwo",       "lbAUCsubTwo"),
    "mpnet_no_offset":          ("lbNLLsubThree",     "lbAUCsubThree"),
    "tscale_mpnet":             ("lbNLLsubFive",      "lbAUCsubFive"),
    "tscale_T3p0":              ("lbNLLsubSix",       "lbAUCsubSix"),
    "tscale_T5p5":              ("lbNLLsubSeven",     "lbAUCsubSeven"),
    "tscale_plus_offset":       ("lbNLLsubEight",     "lbAUCsubEight"),
    "const_0p5":                ("lbNLLconstHalf",    "lbAUCconstHalf"),
    "const_glob_mean":          ("lbNLLconstGM",      "lbAUCconstGM"),
    "subj_mean_lookup":         ("lbNLLsubjMean",     "lbAUCsubjMean"),
    "bm_cond_lookup":           ("lbNLLbmCond",       "lbAUCbmCond"),
    "sub5_blend_subj":          ("lbNLLblendSubj",    "lbAUCblendSubj"),
    "sub5_labeled_shift":       ("lbNLLlabeledShift", "lbAUClabeledShift"),
}

START_MARK = "% -- known scores from prior rounds"
END_MARK = "% ===================================================================="


def fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else r"\TBD"


def load_latest_scores():
    conn = sqlite3.connect(LEDGER)
    rows = conn.execute(
        """
        SELECT model_name, leaderboard_nll, leaderboard_auc
        FROM submissions
        WHERE id IN (
            SELECT MAX(id) FROM submissions GROUP BY model_name
        )
        """
    ).fetchall()
    return {name: (nll, auc) for name, nll, auc in rows}


def render_macros(scores):
    out = []
    for model, (nll_macro, auc_macro) in MACRO_MAP.items():
        nll, auc = scores.get(model, (None, None))
        out.append(rf"\newcommand{{\{nll_macro}}}{{{fmt(nll)}}}"
                   rf"\newcommand{{\{auc_macro}}}{{{fmt(auc)}}}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-place", action="store_true",
                   help="rewrite the macro block in docs/ablations_section.tex")
    args = p.parse_args()

    scores = load_latest_scores()
    block = render_macros(scores)

    if not args.in_place:
        print(block)
        return

    if not TEX.exists():
        sys.exit(f"missing: {TEX}")
    txt = TEX.read_text()
    # Replace from "% -- known scores from prior rounds" up to (exclusive of)
    # the closing "% =====..." line.
    pattern = re.compile(
        rf"({re.escape(START_MARK)}.*?)({re.escape(END_MARK)})",
        re.DOTALL,
    )
    m = pattern.search(txt)
    if not m:
        sys.exit("could not find macro block markers in ablations_section.tex")
    replacement = f"{START_MARK} -----------------------\n{block}\n{END_MARK}"
    new_txt = txt[: m.start()] + replacement + txt[m.end():]
    TEX.write_text(new_txt)
    print(f"updated {TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
