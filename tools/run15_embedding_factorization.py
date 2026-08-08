#!/usr/bin/env python3
"""Run 15: factorize SmolLM2's tied input embedding / LM head.

The tied vocabulary matrix is ~21% of SmolLM2-135M parameters. Therefore a
10x Q4-relative weight target is mathematically impossible if that matrix remains
unchanged, even if all decoder parameters were free.

We test the optimal Frobenius rank-r factorization E ~= A B using the 576x576
Gram eigensystem, then evaluate both FP32-factor ceilings and a representation-
matched Q4_GROUP64(A,B) contract:

  embedding(token) = A[token] B
  logits(h)         = (h B^T) A^T

Quality is measured three ways on held-out WikiText-2 tokens:
  1. occurrence-weighted input-embedding vector NMSE;
  2. head-only next-token NLL using the teacher's original final hidden states;
  3. integrated partial-model NLL with factorized input embeddings propagated
     through the untouched 30-layer decoder and the factorized output head.

No calibration activations are used to fit the factors; the decomposition uses
only the pretrained tied vocabulary matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

GROUP = 64


def q4_group64(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    w = w.detach().float().contiguous()
    rows, cols = w.shape
    out = torch.empty_like(w)
    nbytes = 0
    for s in range(0, cols, GROUP):
        e = min(cols, s + GROUP)
        x = w[:, s:e]
        scale = x.abs().amax(dim=1, keepdim=True) / 7.0
        scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
        qi = torch.round(x / scale).clamp_(-8, 7)
        out[:, s:e] = qi * scale
        nbytes += rows * (math.ceil((e - s) / 2) + 2)
    return out, nbytes


def q4_bytes(rows: int, cols: int) -> int:
    total = 0
    for s in range(0, cols, GROUP):
        width = min(GROUP, cols - s)
        total += rows * (math.ceil(width / 2) + 2)
    return total


def make_factors(e: torch.Tensor, rank: int, basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = basis[:, :rank]
    a = e @ v
    b = v.T.contiguous()
    return a.contiguous(), b


def factor_logits(h: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # h [..., D], b [R,D], a [V,R]
    z = F.linear(h, b)      # [..., R]
    return F.linear(z, a)   # [..., V]


def factor_embed(ids: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.embedding(ids, a) @ b


def ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"))


def ppl(nll_sum: float, n: int) -> float:
    return math.exp(nll_sum / n)


def eval_reference_and_head_only(
    model,
    ids: torch.Tensor,
    context: int,
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ref_nll = 0.0
    total = 0
    fact_nll = {k: 0.0 for k in factors}
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, context):
            seq = ids[start:start + context + 1]
            if seq.numel() < 2:
                break
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            h = model.model(input_ids=x, use_cache=False).last_hidden_state.float()
            ref_logits = F.linear(h, model.lm_head.weight.float())
            ref_nll += ce_sum(ref_logits, y)
            total += int(y.numel())
            for name, (a, b) in factors.items():
                fact_nll[name] += ce_sum(factor_logits(h, a, b), y)
    ref = {"predicted_tokens": total, "nll": ref_nll / total, "ppl": ppl(ref_nll, total)}
    out = {}
    for name, nll_sum in fact_nll.items():
        nll = nll_sum / total
        out[name] = {
            "nll": nll,
            "ppl": math.exp(nll),
            "delta_nll_vs_reference": nll - ref["nll"],
            "ppl_ratio_vs_reference": math.exp(nll - ref["nll"]),
        }
    return ref, out


def eval_integrated(
    model,
    ids: torch.Tensor,
    context: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, Any]:
    nll_sum = 0.0
    total = 0
    with torch.inference_mode():
        for start in range(0, ids.numel() - 1, context):
            seq = ids[start:start + context + 1]
            if seq.numel() < 2:
                break
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            emb = factor_embed(x, a, b)
            h = model.model(inputs_embeds=emb, use_cache=False).last_hidden_state.float()
            logits = factor_logits(h, a, b)
            nll_sum += ce_sum(logits, y)
            total += int(y.numel())
    nll = nll_sum / total
    return {"predicted_tokens": total, "nll": nll, "ppl": math.exp(nll)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--ranks", default="64,128,192,256")
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

    embed = model.model.embed_tokens.weight.detach().float().cpu().contiguous()
    head = model.lm_head.weight.detach().float().cpu().contiguous()
    tied_storage = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    max_diff = float((embed - head).abs().max())
    if max_diff > 1e-7:
        raise RuntimeError(f"Input embedding and lm_head are not numerically tied: max diff {max_diff}")

    vocab, hidden = embed.shape
    exact_model_params = sum(p.numel() for p in model.parameters())
    embedding_params = embed.numel()
    embedding_param_fraction = embedding_params / exact_model_params

    text = args.evaluation.read_text(errors="replace")
    all_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = min(all_ids.numel(), args.eval_tokens + 1)
    ids = all_ids[:need].contiguous()
    if ids.numel() < 2:
        raise RuntimeError("Evaluation token stream empty")

    # Optimal right subspace for E under Frobenius reconstruction.
    gram = embed.T @ embed
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    basis = evecs[:, order].contiguous()
    total_energy = float(evals.sum())

    ranks = [int(x) for x in args.ranks.split(",")]
    fp_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    q4_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    config_meta: dict[str, dict[str, Any]] = {}
    baseline_q4 = q4_bytes(vocab, hidden)
    baseline_fp16 = embed.numel() * 2

    occurrence_embed = embed.index_select(0, ids)
    for rank in ranks:
        if rank <= 0 or rank > hidden:
            raise ValueError(f"Invalid rank {rank}")
        a, b = make_factors(embed, rank, basis)
        aq, abytes = q4_group64(a)
        bq, bbytes = q4_group64(b)
        fp_factors[f"r{rank}"] = (a, b)
        q4_factors[f"r{rank}"] = (aq, bq)

        fp_occ = factor_embed(ids, a, b)
        q_occ = factor_embed(ids, aq, bq)
        occ_den = float(occurrence_embed.square().sum())
        fp_nmse = float((fp_occ - occurrence_embed).square().sum()) / max(occ_den, 1e-30)
        q_nmse = float((q_occ - occurrence_embed).square().sum()) / max(occ_den, 1e-30)
        factor_bytes = abytes + bbytes
        fp_factor_bytes = (a.numel() + b.numel()) * 2
        retained = float(evals[:rank].sum()) / max(total_energy, 1e-30)
        config_meta[f"r{rank}"] = {
            "rank": rank,
            "spectral_energy_retained": retained,
            "fp16_factor_bytes": fp_factor_bytes,
            "fp16_structural_reduction_vs_tied_fp16_x": baseline_fp16 / fp_factor_bytes,
            "q4_factor_bytes": factor_bytes,
            "q4_reduction_vs_tied_q4_group64_x": baseline_q4 / factor_bytes,
            "occurrence_weighted_input_embedding_nmse": {
                "fp32_factors": fp_nmse,
                "q4_factors": q_nmse,
            },
        }

    t0 = time.perf_counter()
    reference, head_fp = eval_reference_and_head_only(model, ids, args.context, fp_factors)
    _, head_q4 = eval_reference_and_head_only(model, ids, args.context, q4_factors)

    configs = []
    for rank in ranks:
        key = f"r{rank}"
        fp_integrated = eval_integrated(model, ids, args.context, *fp_factors[key])
        q_integrated = eval_integrated(model, ids, args.context, *q4_factors[key])
        for row in (fp_integrated, q_integrated):
            row["delta_nll_vs_reference"] = row["nll"] - reference["nll"]
            row["ppl_ratio_vs_reference"] = math.exp(row["delta_nll_vs_reference"])
        row = dict(config_meta[key])
        row["head_only"] = {"fp32_factors": head_fp[key], "q4_factors": head_q4[key]}
        row["integrated_input_and_head"] = {
            "fp32_factors": fp_integrated,
            "q4_factors": q_integrated,
        }
        configs.append(row)

    passing = []
    borderline = []
    for c in configs:
        red = c["q4_reduction_vs_tied_q4_group64_x"]
        emb_nmse = c["occurrence_weighted_input_embedding_nmse"]["q4_factors"]
        head_ratio = c["head_only"]["q4_factors"]["ppl_ratio_vs_reference"]
        full_ratio = c["integrated_input_and_head"]["q4_factors"]["ppl_ratio_vs_reference"]
        if red >= 4.0 and emb_nmse <= 0.05 and head_ratio <= 1.05 and full_ratio <= 1.10:
            passing.append(c["rank"])
        elif red >= 3.0 and emb_nmse <= 0.10 and head_ratio <= 1.10 and full_ratio <= 1.25:
            borderline.append(c["rank"])

    decision = "pass_component_gate" if passing else ("borderline_component_gate" if borderline else "fail_component_gate")
    out = {
        "run": 15,
        "kind": "tied_embedding_and_lm_head_factorization_diagnostic",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "tied_storage": tied_storage,
        "input_head_max_abs_difference": max_diff,
        "geometry": {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "exact_unique_model_parameter_count": exact_model_params,
            "tied_embedding_parameter_count": embedding_params,
            "tied_embedding_fraction_of_unique_parameters": embedding_param_fraction,
            "unchanged_embedding_absolute_weight_reduction_ceiling_x_if_all_other_parameters_free": 1.0 / embedding_param_fraction,
        },
        "evaluation": {
            "file": str(args.evaluation),
            "token_stream_tokens": int(ids.numel()),
            "predicted_tokens": reference["predicted_tokens"],
            "context": args.context,
            "reference": reference,
            "note": "Fixed leading WikiText-2 test slice; not the full standard-corpus promotion benchmark. Factors are fit from model weights only, not evaluation activations."
        },
        "representation": {
            "factorization": "E ~= A B from eig(E^T E); same A/B used for input embedding and output head",
            "packed": "Q4_GROUP64 A + Q4_GROUP64 B; no dense E shadow counted",
            "runtime_math": "embedding=A[token]@B; logits=(hidden@B.T)@A.T",
            "baseline": "single tied E stored as Q4_GROUP64"
        },
        "configs": configs,
        "precommitted_component_gate": {
            "pass": "Q4 factor reduction >=4x; occurrence-weighted embedding NMSE <=0.05; head-only PPL ratio <=1.05; integrated input+head PPL ratio <=1.10",
            "borderline": "reduction >=3x; embedding NMSE <=0.10; head-only PPL ratio <=1.10; integrated ratio <=1.25"
        },
        "passing_ranks": passing,
        "borderline_ranks": borderline,
        "decision": decision,
        "wall_seconds": time.perf_counter() - t0,
        "claim_boundary": (
            "Real-pretrained tied embedding/head component diagnostic on a fixed WikiText-2 test slice. "
            "Decoder weights are unchanged. No full-corpus standard PPL promotion, packed native kernel, RSS, VRAM, "
            "or whole-model compression claim. Q4_GROUP64 is the component byte baseline, not llama.cpp Q4_K_M."
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "geometry": out["geometry"],
        "reference": reference,
        "configs": [{
            "rank": c["rank"],
            "q4_reduction_x": c["q4_reduction_vs_tied_q4_group64_x"],
            "q4_embedding_nmse": c["occurrence_weighted_input_embedding_nmse"]["q4_factors"],
            "head_q4_ppl_ratio": c["head_only"]["q4_factors"]["ppl_ratio_vs_reference"],
            "integrated_q4_ppl_ratio": c["integrated_input_and_head"]["q4_factors"]["ppl_ratio_vs_reference"]
        } for c in configs]
    }, indent=2))


if __name__ == "__main__":
    main()
