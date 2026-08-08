#!/usr/bin/env python3
"""Run 17: direct-packed product quantization for the tied vocabulary matrix.

The SmolLM2 tied embedding/LM-head matrix is ~21% of unique parameters and by
itself exceeds the entire measured 10x-vs-Q4 file budget when stored in ordinary
Q4. Run 15 showed that global low-rank factorization destroys language-model
quality. Run 17 preserves full-rank geometry by encoding each token vector as a
composition of subspace centroids.

Representation for token t:
    E[t] ~= norm[t] * concat(C_0[code[t,0]], ..., C_{M-1}[code[t,M-1]])

where every code is uint8, every centroid is FP16, and norm[t] is FP16.
No dense E shadow is part of the representation.

Direct input lookup:
    gather one centroid/subspace, concatenate, multiply by norm[t].

Direct output logits for hidden h:
    for each subspace s compute table_s[k] = dot(h_s, C_s[k]);
    logit[t] = norm[t] * sum_s table_s[code[t,s]].

The script verifies this direct logit math against a separately reconstructed
dense reference before evaluating held-out language-model quality.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

GROUP = 64
K = 256


def q4_bytes(rows: int, cols: int) -> int:
    total = 0
    for s in range(0, cols, GROUP):
        width = min(GROUP, cols - s)
        total += rows * (math.ceil(width / 2) + 2)
    return total


def deterministic_kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic Lloyd k-means over one embedding subspace.

    x is [vocab, d] on CPU float32. Initialization uses a seeded uniform sample.
    The full vocabulary participates in every Lloyd iteration; no corpus labels or
    activation data are used.
    """
    n, d = x.shape
    if n < k:
        raise ValueError("vocabulary smaller than codebook")
    gen = torch.Generator().manual_seed(seed)
    init = torch.randperm(n, generator=gen)[:k]
    c = x.index_select(0, init).clone()
    x2 = x.square().sum(dim=1, keepdim=True)
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dist = x2 + c.square().sum(dim=1)[None, :] - 2.0 * (x @ c.T)
        labels = dist.argmin(dim=1)
        sums = torch.zeros_like(c)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k)
        nonempty = counts > 0
        c[nonempty] = sums[nonempty] / counts[nonempty, None]
    return c.contiguous(), labels.contiguous()


def fit_pq(e: torch.Tensor, subdim: int, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    vocab, hidden = e.shape
    if hidden % subdim:
        raise ValueError(f"hidden={hidden} not divisible by subdim={subdim}")
    m = hidden // subdim
    norm = e.norm(dim=1).clamp_min(1e-12)
    unit = e / norm[:, None]
    codebooks = []
    codes = torch.empty((vocab, m), dtype=torch.uint8)
    occupancy = []
    for s in range(m):
        xs = unit[:, s * subdim:(s + 1) * subdim].contiguous()
        c, lab = deterministic_kmeans(xs, K, iters, seed=17000 + subdim * 101 + s)
        codebooks.append(c)
        codes[:, s] = lab.to(torch.uint8)
        cnt = torch.bincount(lab, minlength=K)
        occupancy.append({
            "subspace": s,
            "nonempty_centroids": int((cnt > 0).sum()),
            "min_nonzero_count": int(cnt[cnt > 0].min()) if (cnt > 0).any() else 0,
            "max_count": int(cnt.max()),
        })
    cb = torch.stack(codebooks, dim=0).contiguous()
    return cb, codes, norm, {"subspaces": m, "occupancy": occupancy}


def reconstruct(codebooks: torch.Tensor, codes: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    vocab, m = codes.shape
    parts = []
    for s in range(m):
        parts.append(codebooks[s].index_select(0, codes[:, s].long()))
    unit = torch.cat(parts, dim=1)
    return unit * norm[:, None]


def direct_logits(h: torch.Tensor, codebooks: torch.Tensor, codes: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    """Direct semantic packed-head math, no dense vocabulary matrix required."""
    hidden = h.shape[-1]
    m, k, subdim = codebooks.shape
    if m * subdim != hidden:
        raise ValueError("hidden dimension mismatch")
    flat = h.reshape(-1, hidden)
    out = torch.zeros((flat.shape[0], codes.shape[0]), dtype=flat.dtype)
    for s in range(m):
        hs = flat[:, s * subdim:(s + 1) * subdim]
        table = hs @ codebooks[s].T
        idx = codes[:, s].long()[None, :].expand(flat.shape[0], -1)
        out += torch.gather(table, 1, idx)
    out *= norm[None, :]
    return out.reshape(*h.shape[:-1], codes.shape[0])


def direct_embed(ids: torch.Tensor, codebooks: torch.Tensor, codes: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    flat_ids = ids.reshape(-1).long()
    m, _, subdim = codebooks.shape
    parts = []
    selected_codes = codes.index_select(0, flat_ids)
    for s in range(m):
        parts.append(codebooks[s].index_select(0, selected_codes[:, s].long()))
    unit = torch.cat(parts, dim=1)
    emb = unit * norm.index_select(0, flat_ids)[:, None]
    return emb.reshape(*ids.shape, m * subdim)


def ce_sum(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))


def eval_reference(model, ids: torch.Tensor, context: int) -> dict[str, float]:
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, context):
            seq = ids[start:start + context + 1]
            if seq.numel() < 2:
                break
            x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
            logits = model(input_ids=x, use_cache=False).logits.float()
            nll += ce_sum(logits, y)
            total += int(y.numel())
    mean = nll / total
    return {"predicted_tokens": total, "nll": mean, "ppl": math.exp(mean)}


def eval_head_only(model, ids: torch.Tensor, context: int, cb, codes, norm) -> dict[str, float]:
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, context):
            seq = ids[start:start + context + 1]
            if seq.numel() < 2:
                break
            x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
            h = model.model(input_ids=x, use_cache=False).last_hidden_state.float()
            logits = direct_logits(h, cb, codes, norm)
            nll += ce_sum(logits, y)
            total += int(y.numel())
    mean = nll / total
    return {"predicted_tokens": total, "nll": mean, "ppl": math.exp(mean)}


def eval_integrated(model, ids: torch.Tensor, context: int, cb, codes, norm) -> dict[str, float]:
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, context):
            seq = ids[start:start + context + 1]
            if seq.numel() < 2:
                break
            x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
            emb = direct_embed(x, cb, codes, norm)
            h = model.model(inputs_embeds=emb, use_cache=False).last_hidden_state.float()
            logits = direct_logits(h, cb, codes, norm)
            nll += ce_sum(logits, y)
            total += int(y.numel())
    mean = nll / total
    return {"predicted_tokens": total, "nll": mean, "ppl": math.exp(mean)}


def add_relative(row: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    row = dict(row)
    row["delta_nll_vs_reference"] = row["nll"] - ref["nll"]
    row["ppl_ratio_vs_reference"] = math.exp(row["delta_nll_vs_reference"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--subdims", default="8,12,16,24,32")
    ap.add_argument("--kmeans-iters", type=int, default=6)
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    ).eval()

    e = model.model.embed_tokens.weight.detach().float().cpu().contiguous()
    if float((e - model.lm_head.weight.detach().float().cpu()).abs().max()) > 1e-7:
        raise RuntimeError("embedding and LM head are not tied numerically")
    vocab, hidden = e.shape
    baseline_q4 = q4_bytes(vocab, hidden)

    text = args.evaluation.read_text(errors="replace")
    all_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = all_ids[:min(all_ids.numel(), args.eval_tokens + 1)].contiguous()
    if ids.numel() < 2:
        raise RuntimeError("evaluation stream is empty")

    ref = eval_reference(model, ids, args.context)
    occurrence_ref = e.index_select(0, ids)
    occ_den = float(occurrence_ref.square().sum())
    configs = []
    t0 = time.perf_counter()

    for subdim in [int(x) for x in args.subdims.split(",")]:
        cb32, codes, norm32, fit_meta = fit_pq(e, subdim, args.kmeans_iters)
        cb = cb32.half().float().contiguous()
        norm = norm32.half().float().contiguous()
        dense = reconstruct(cb, codes, norm)

        probe_h = torch.randn((3, hidden), generator=torch.Generator().manual_seed(17017 + subdim))
        dense_logits = probe_h @ dense.T
        packed_logits = direct_logits(probe_h, cb, codes, norm)
        semantic_max_abs = float((dense_logits - packed_logits).abs().max())

        occ_hat = dense.index_select(0, ids)
        occ_nmse = float((occ_hat - occurrence_ref).square().sum()) / max(occ_den, 1e-30)
        all_nmse = float((dense - e).square().sum()) / max(float(e.square().sum()), 1e-30)

        head = add_relative(eval_head_only(model, ids, args.context, cb, codes, norm), ref)
        integrated = add_relative(eval_integrated(model, ids, args.context, cb, codes, norm), ref)

        m = hidden // subdim
        codebook_bytes = m * K * subdim * 2
        code_bytes = vocab * m
        norm_bytes = vocab * 2
        total_bytes = codebook_bytes + code_bytes + norm_bytes
        configs.append({
            "subdim": subdim,
            "subspaces": m,
            "centroids_per_subspace": K,
            "bits_per_subspace_code": 8,
            "bytes": {
                "fp16_codebooks": codebook_bytes,
                "uint8_token_codes": code_bytes,
                "fp16_token_norms": norm_bytes,
                "total": total_bytes,
                "baseline_tied_q4_group64": baseline_q4,
                "reduction_vs_tied_q4_group64_x": baseline_q4 / total_bytes,
                "effective_bits_per_original_embedding_weight": total_bytes * 8 / e.numel(),
            },
            "fit": fit_meta,
            "semantic_direct_head_max_abs_error_vs_dense_decoded": semantic_max_abs,
            "reconstruction": {
                "whole_vocab_weight_nmse": all_nmse,
                "heldout_occurrence_weighted_embedding_nmse": occ_nmse,
            },
            "head_only": head,
            "integrated_input_and_head": integrated,
        })

    passing = []
    borderline = []
    for c in configs:
        red = c["bytes"]["reduction_vs_tied_q4_group64_x"]
        nmse = c["reconstruction"]["heldout_occurrence_weighted_embedding_nmse"]
        head = c["head_only"]["ppl_ratio_vs_reference"]
        integ = c["integrated_input_and_head"]["ppl_ratio_vs_reference"]
        sem = c["semantic_direct_head_max_abs_error_vs_dense_decoded"]
        if red >= 5.0 and nmse <= 0.05 and head <= 1.05 and integ <= 1.10 and sem <= 1e-4:
            passing.append(c["subdim"])
        elif red >= 4.0 and nmse <= 0.10 and head <= 1.15 and integ <= 1.50 and sem <= 1e-4:
            borderline.append(c["subdim"])

    decision = "pass_component_gate" if passing else ("borderline_component_gate" if borderline else "fail_component_gate")
    out = {
        "run": 17,
        "kind": "direct_packed_tied_vocabulary_product_quantization",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "geometry": {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "tied_q4_group64_baseline_bytes": baseline_q4,
        },
        "representation": {
            "token_norm": "FP16 one per vocabulary row",
            "subspace_codes": "uint8 one centroid index per token/subspace",
            "codebooks": "FP16 256 centroids per fixed contiguous subspace",
            "dense_shadow_counted": False,
            "input_runtime": "centroid gathers + concatenate + token norm",
            "head_runtime": "256 dot products/subspace + code gathers/sums + token norm; no dense E reconstruction required",
        },
        "fit": {
            "uses_model_weights_only": True,
            "uses_evaluation_activations_or_labels": False,
            "kmeans_iters": args.kmeans_iters,
            "distance": "Euclidean on globally row-normalized token vectors",
        },
        "evaluation": {
            "file": str(args.evaluation),
            "token_stream_tokens": int(ids.numel()),
            "predicted_tokens": ref["predicted_tokens"],
            "context": args.context,
            "reference": ref,
            "note": "fixed leading WikiText-2 test slice; not full-corpus promotion benchmark",
        },
        "configs": configs,
        "precommitted_component_gate": {
            "pass": ">=5x tied-Q4 reduction; occurrence embedding NMSE <=0.05; head-only PPL ratio <=1.05; integrated <=1.10; direct packed math error <=1e-4",
            "borderline": ">=4x reduction; NMSE <=0.10; head-only <=1.15; integrated <=1.50; direct packed math error <=1e-4",
        },
        "passing_subdims": passing,
        "borderline_subdims": borderline,
        "decision": decision,
        "wall_seconds": time.perf_counter() - t0,
        "claim_boundary": (
            "Real-pretrained tied-vocabulary component diagnostic on a fixed WikiText-2 slice. Codebook/codes/norm bytes are exact for the stated representation, and direct packed output math is semantically checked, but no native optimized kernel, process RSS, VRAM or whole-model LARC claim is made. Q4_GROUP64 is the component baseline, not llama.cpp Q4_K_M."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "reference": ref,
        "configs": [{
            "subdim": c["subdim"],
            "reduction_x": c["bytes"]["reduction_vs_tied_q4_group64_x"],
            "effective_bpw": c["bytes"]["effective_bits_per_original_embedding_weight"],
            "occ_nmse": c["reconstruction"]["heldout_occurrence_weighted_embedding_nmse"],
            "head_ppl_ratio": c["head_only"]["ppl_ratio_vs_reference"],
            "integrated_ppl_ratio": c["integrated_input_and_head"]["ppl_ratio_vs_reference"],
            "semantic_error": c["semantic_direct_head_max_abs_error_vs_dense_decoded"],
        } for c in configs]
    }, indent=2))


if __name__ == "__main__":
    main()
