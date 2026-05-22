"""llm_judge_qwen — local LLM-as-judge per worked-example §3.5.

Loads Qwen2.5-0.5B-Instruct (a small instruction-tuned LLM that fits
the Codabench T4 sandbox easily), formats the four input fields into
the JUDGE template prescribed by the worked example, runs ONE forward
pass over the prompt, and reads the next-token log-probabilities of
" yes" and " no" to recover a calibrated p(correct).

This is the F5 row of our coverage table moving from `×` to `✓`:
previously we excluded local LLM judges on per-tier compute grounds,
which is true for a 7B-class judge (~14 GB GPU memory) but not for
a 0.5B-class one (~1 GB in bf16). The §3.5 spec itself notes that
"smaller backbone" is a way to fit the per-round budget.

Design choices:
  * Raw text JUDGE template exactly per §3.5 (not chat-templated),
    so this submission is a literal replication of the worked-example
    pseudocode rather than a chat-template optimization.
  * Read next-token log-probs of YES_ID / NO_ID and renormalize
    over {yes, no} (closed-form calibrated binary probability).
  * bf16 on CUDA, fp32 on CPU.
  * No use of labeled[] for in-context calibration (that's the
    "stronger variant" §3.5 calls out; see notes in manifest).

The labeled argument of predict() is unused in this baseline; the
platform falls back to random K=5 per category, identical to every
prior submission in our ledger that omits labeling.py.
"""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if _DEVICE == "cuda" else torch.float32

TOK = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=_DTYPE,
).to(_DEVICE)
MODEL.eval()

YES_ID = TOK.encode(" yes", add_special_tokens=False)[-1]
NO_ID = TOK.encode(" no", add_special_tokens=False)[-1]

JUDGE = """You will see a description of an AI subject and an
evaluation item. Decide whether the subject would answer the item
correctly. Reply with a single token: yes or no.

Benchmark: {benchmark}
Condition: {condition}
Subject: {subject_content}
Item: {item_content}
Answer:"""


def _safe(input: dict, key: str) -> str:
    v = input.get(key, "")
    if v is None:
        return ""
    return str(v)


def predict(input, labeled=None):
    prompt = JUDGE.format(
        benchmark=_safe(input, "benchmark"),
        condition=_safe(input, "condition"),
        subject_content=_safe(input, "subject_content"),
        item_content=_safe(input, "item_content"),
    )
    try:
        ids = TOK(prompt, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            out = MODEL(**ids)
            logits = out.logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            p_yes = math.exp(lp[YES_ID].item())
            p_no = math.exp(lp[NO_ID].item())
        denom = p_yes + p_no
        if denom <= 0.0 or not math.isfinite(denom):
            return 0.5
        return float(p_yes / denom)
    except Exception:
        return 0.5
