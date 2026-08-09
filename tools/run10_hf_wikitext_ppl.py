#!/usr/bin/env python3
"""Deterministic WikiText-2 perplexity for FP16 and packed AutoRound models.

Every representation uses exactly the same token IDs and non-overlapping context
boundaries. Equal-length windows are batched only for throughput; batching does
not change which next-token predictions enter the loss. Absolute PPL is not
asserted to be bit-for-bit identical to llama.cpp's evaluator; Run 9 owns the
llama.cpp Q4_K_M/Q2_K baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def batch_loss(model, xs: list[torch.Tensor], ys: list[torch.Tensor]) -> tuple[float, int]:
    if not xs:
        return 0.0, 0
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    logits = model(input_ids=x, use_cache=False).logits.float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="sum",
    )
    return float(loss), int(y.numel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch-windows", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.batch_windows < 1:
        raise ValueError("--batch-windows must be >=1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer_name = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    text = args.corpus.read_text(errors="replace")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < 2:
        raise RuntimeError("Corpus tokenization is empty")

    full_x: list[torch.Tensor] = []
    full_y: list[torch.Tensor] = []
    tail: tuple[torch.Tensor, torch.Tensor] | None = None
    chunks = 0
    for start in range(0, ids.numel() - 1, args.context):
        seq = ids[start:start + args.context + 1]
        if seq.numel() < 2:
            break
        x, y = seq[:-1], seq[1:]
        chunks += 1
        if x.numel() == args.context:
            full_x.append(x)
            full_y.append(y)
        else:
            # At most one final short token-stream window exists.
            tail = (x, y)

    total_nll = 0.0
    total_pred = 0
    batches = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(full_x), args.batch_windows):
            loss, n = batch_loss(model, full_x[i:i + args.batch_windows], full_y[i:i + args.batch_windows])
            total_nll += loss
            total_pred += n
            batches += 1
        if tail is not None:
            loss, n = batch_loss(model, [tail[0]], [tail[1]])
            total_nll += loss
            total_pred += n
            batches += 1

    expected_pred = int(ids.numel() - 1)
    if total_pred != expected_pred:
        raise RuntimeError(f"Prediction accounting mismatch: measured {total_pred}, expected {expected_pred}")

    mean_nll = total_nll / total_pred
    out = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "corpus": str(args.corpus),
        "context": args.context,
        "batch_windows": args.batch_windows,
        "tokenized_corpus_tokens": int(ids.numel()),
        "predicted_tokens": total_pred,
        "chunks": chunks,
        "forward_batches": batches,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "wall_seconds": time.perf_counter() - t0,
        "method": (
            "non-overlapping HF token-stream windows; add_special_tokens=False; next-token CE within every window; "
            "equal-length windows batched without changing token IDs or context boundaries"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
