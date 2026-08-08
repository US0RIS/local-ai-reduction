#!/usr/bin/env python3
"""Run 7: segmented cross-layer shared-basis compression on a real LLM.

Run 6 falsified two universal assumptions on SmolLM2-135M:
  * low activation rank is not uniformly available across operators/layers;
  * several logical decoder layers cannot simply alias one physical block after
    light recovery while preserving quality.

Run 7 therefore preserves depth-specific functions. For a like operator in a
contiguous layer group, each matrix is represented as

    W_i ~= B_g C_i

where B_g [out,r] is one physical basis shared across the group and C_i [r,in]
is unique to logical layer i. B_g is fitted from real calibration operator
outputs, not weight SVD alone.

Evidence is intentionally split:
  1. FP32 structural fit: FP32 activations/weights -> FP32 factors.
  2. Deployment-matched fit: row-Q4 activations/weights -> Q4_GROUP64 factors.

The Q4 rank schedule is selected ONLY from the deployment-matched held-out
operator errors, so quality/memory representations cannot silently diverge.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from run6_real_model_falsification import TEXT, find_projection, get_layers, nll

SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj")
DEFAULT_RANKS = (16, 32, 64, 96, 128, 192, 256, 384, 512)
# Chosen before Run-7 model results are observed.
GATES = {
    "strict": {"mean_nmse": 0.03, "max_nmse": 0.05},
    "balanced": {"mean_nmse": 0.05, "max_nmse": 0.10},
    "aggressive": {"mean_nmse": 0.10, "max_nmse": 0.20},
}


def parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def contiguous_groups(n: int, size: int) -> list[list[int]]:
    return [
        list(range(i, min(n, i + size)))
        for i in range(0, n, size)
        if min(n, i + size) - i >= 2
    ]


def collect_site_inputs(model, ids: torch.Tensor, site: str) -> dict[int, torch.Tensor]:
    layers = get_layers(model)
    out: dict[int, torch.Tensor] = {}
    handles = []
    for li, layer in enumerate(layers):
        try:
            mod = find_projection(layer, site)
        except AttributeError:
            continue

        def hook(_m, inp, _out, li=li):
            out[li] = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()

        handles.append(mod.register_forward_hook(hook))
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return out


def q4_group64_tensor(w: torch.Tensor, group: int = 64) -> torch.Tensor:
    x = w.detach().float()
    if x.ndim != 2:
        return x.half().float()
    y = torch.empty_like(x)
    for s in range(0, x.shape[1], group):
        a = x[:, s : s + group]
        pos = a.amax(1).clamp_min(0) / 7.0
        neg = (-a.amin(1)).clamp_min(0) / 8.0
        sc = torch.maximum(pos, neg).clamp_min(1e-8).half().float()
        y[:, s : s + group] = torch.round(a / sc[:, None]).clamp(-8, 7) * sc[:, None]
    return y


def row_q4_tensor(w: torch.Tensor) -> torch.Tensor:
    x = w.detach().float()
    if x.ndim != 2:
        return x.half().float()
    pos = x.amax(1).clamp_min(0) / 7.0
    neg = (-x.amin(1)).clamp_min(0) / 8.0
    sc = torch.maximum(pos, neg).clamp_min(1e-8).half().float()
    return torch.round(x / sc[:, None]).clamp(-8, 7) * sc[:, None]


def project_model_row_q4_(m) -> None:
    with torch.no_grad():
        seen = set()
        for p in m.parameters():
            ptr = p.untyped_storage().data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            p.copy_(row_q4_tensor(p))


def q4_group64_bytes(rows: int, cols: int, group: int = 64) -> int:
    return rows * (((cols + 1) // 2) + math.ceil(cols / group) * 2)


def row_q4_bytes(rows: int, cols: int) -> int:
    return rows * (((cols + 1) // 2) + 2)


def modeled_bytes(model, factor_param_ids=frozenset()) -> int:
    """Modeled weight bytes with ordinary matrices row-Q4 and factor matrices group64-Q4."""
    total = 0
    seen = set()
    for p in model.parameters():
        ptr = p.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if p.ndim == 2:
            total += (
                q4_group64_bytes(p.shape[0], p.shape[1])
                if id(p) in factor_param_ids
                else row_q4_bytes(p.shape[0], p.shape[1])
            )
        else:
            total += p.numel() * 2
    return total


class SharedBasisLinear(nn.Module):
    """Two GEMVs, sharing the output basis Parameter across logical layers."""

    def __init__(self, basis_param: nn.Parameter, coeff: torch.Tensor, bias=None):
        super().__init__()
        self.basis = basis_param
        self.coeff = nn.Parameter(coeff, requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    def forward(self, x):
        return F.linear(F.linear(x, self.coeff), self.basis, self.bias)


def factor_errors(
    B: torch.Tensor,
    coeffs: dict[int, torch.Tensor],
    ws: dict[int, torch.Tensor],
    eval_inputs: dict[int, torch.Tensor],
) -> tuple[list[float], float]:
    vals = []
    num = 0.0
    den = 0.0
    for li, w in ws.items():
        x = eval_inputs[li]
        y = x @ w.t()
        pred = (x @ coeffs[li].t()) @ B.t()
        e = float(((pred - y) ** 2).sum())
        p = float((y**2).sum().clamp_min(1e-30))
        vals.append(e / p)
        num += e
        den += p
    return vals, num / den


def fit_group(
    site: str,
    layers,
    cal_inputs: dict[int, torch.Tensor],
    eval_inputs: dict[int, torch.Tensor],
    group: list[int],
    ranks: tuple[int, ...],
):
    """Fit shared output bases from one representation (FP32 OR row-Q4)."""
    ws: dict[int, torch.Tensor] = {}
    yc = []
    for li in group:
        w = find_projection(layers[li], site).weight.detach().float().cpu()
        ws[li] = w
        yc.append(cal_inputs[li] @ w.t())
    ycat = torch.cat(yc, 0)
    out_dim = next(iter(ws.values())).shape[0]
    max_req = min(max(ranks), out_dim, ycat.shape[0])
    q = min(max_req + 16, out_dim, ycat.shape[0])
    # Randomized PCA is substantially cheaper than a full SVD for MLP outputs.
    _, _, v = torch.pca_lowrank(ycat, q=q, center=False, niter=3)

    records = []
    decomps = {}
    for req in ranks:
        r = min(req, v.shape[1], out_dim)
        if r in decomps:
            continue
        B = v[:, :r].contiguous()  # [out,r]
        coeffs = {li: (B.t() @ w).contiguous() for li, w in ws.items()}
        layer_nmse, aggregate = factor_errors(B, coeffs, ws, eval_inputs)

        # Deployment factor error after the exact Q4_GROUP64 arithmetic intended for storage.
        Bq = q4_group64_tensor(B)
        Cq = {li: q4_group64_tensor(c) for li, c in coeffs.items()}
        qlayer_nmse, qaggregate = factor_errors(Bq, Cq, ws, eval_inputs)

        orig = sum(w.numel() for w in ws.values())
        comp = B.numel() + sum(c.numel() for c in coeffs.values())
        q4_orig = sum(row_q4_bytes(w.shape[0], w.shape[1]) for w in ws.values())
        q4_comp = q4_group64_bytes(B.shape[0], B.shape[1]) + sum(
            q4_group64_bytes(c.shape[0], c.shape[1]) for c in coeffs.values()
        )
        rec = {
            "requested_rank": req,
            "effective_rank": r,
            "group": group,
            "site": site,
            "mean_layer_nmse": statistics.mean(layer_nmse),
            "max_layer_nmse": max(layer_nmse),
            "aggregate_nmse": aggregate,
            "min_layer_nmse": min(layer_nmse),
            "q4factor_mean_layer_nmse": statistics.mean(qlayer_nmse),
            "q4factor_max_layer_nmse": max(qlayer_nmse),
            "q4factor_aggregate_nmse": qaggregate,
            "original_elements": orig,
            "compressed_elements": comp,
            "element_reduction_x": orig / comp,
            "rowq4_original_bytes": q4_orig,
            "group64_factor_bytes": q4_comp,
            "modeled_q4_reduction_x": q4_orig / q4_comp,
        }
        records.append(rec)
        decomps[r] = (B, coeffs, rec)
    return records, decomps


def choose_rank(records: list[dict], gate: dict) -> dict | None:
    """Select from post-factor-quantization error, not FP32 fit error."""
    viable = [
        r
        for r in records
        if r["q4factor_mean_layer_nmse"] <= gate["mean_nmse"]
        and r["q4factor_max_layer_nmse"] <= gate["max_nmse"]
        and r["modeled_q4_reduction_x"] > 1.05
    ]
    return min(viable, key=lambda x: x["effective_rank"]) if viable else None


def replace_with_schedule(model, fitted, schedule, quantize_factors: bool = False):
    layers = get_layers(model)
    factor_ids = set()
    selected = []
    for key, chosen in schedule.items():
        if chosen is None:
            continue
        site, gi_s = key.split(":")
        gi = int(gi_s)
        pack = fitted[site][gi]
        r = chosen["effective_rank"]
        B, coeffs, _ = pack["decomps"][r]
        if quantize_factors:
            B = q4_group64_tensor(B)
        shared = nn.Parameter(B.clone(), requires_grad=False)
        factor_ids.add(id(shared))
        for li in pack["group"]:
            old = find_projection(layers[li], site)
            C = coeffs[li]
            if quantize_factors:
                C = q4_group64_tensor(C)
            new = SharedBasisLinear(shared, C, old.bias)
            factor_ids.add(id(new.coeff))
            # Bias is not a 2D factor and modeled_bytes accounts it as FP16 either way.
            parent = layers[li].self_attn if hasattr(layers[li].self_attn, site) else layers[li].mlp
            setattr(parent, site, new)
        selected.append(
            {
                "site": site,
                "group": pack["group"],
                "rank": r,
                "modeled_q4_reduction_x": chosen["modeled_q4_reduction_x"],
                "fit_mean_nmse": chosen["mean_layer_nmse"],
                "fit_max_nmse": chosen["max_layer_nmse"],
                "q4factor_mean_nmse": chosen["q4factor_mean_layer_nmse"],
                "q4factor_max_nmse": chosen["q4factor_max_layer_nmse"],
            }
        )
    return factor_ids, selected


def fit_all(model, cal, eval_input, groups, ranks):
    layers = get_layers(model)
    fitted = {}
    sweep = {}
    for site in SITES:
        ci = collect_site_inputs(model, cal, site)
        ei = collect_site_inputs(model, eval_input, site)
        fitted[site] = []
        sweep[site] = []
        for group in groups:
            recs, decomps = fit_group(site, layers, ci, ei, group, ranks)
            fitted[site].append({"group": group, "records": recs, "decomps": decomps})
            sweep[site].append({"group": group, "ranks": recs})
        del ci, ei
    return fitted, sweep


def schedule_from_q4(fitted_q4):
    schedules = {}
    for gname, gate in GATES.items():
        sch = {}
        for site in SITES:
            for gi, pack in enumerate(fitted_q4[site]):
                sch[f"{site}:{gi}"] = choose_rank(pack["records"], gate)
        schedules[gname] = sch
    return schedules


def strip_schedule(schedule):
    return {
        k: (
            None
            if v is None
            else {
                x: v[x]
                for x in (
                    "effective_rank",
                    "modeled_q4_reduction_x",
                    "q4factor_mean_layer_nmse",
                    "q4factor_max_layer_nmse",
                )
            }
        )
        for k, v in schedule.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cal-tokens", type=int, default=512)
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/run7_shared_basis_real_model.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(7)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval().cpu()
    layers = get_layers(base)
    ranks = parse_ints(a.ranks)
    ids = tok(TEXT, return_tensors="pt", add_special_tokens=False).input_ids
    need = a.cal_tokens + a.eval_tokens + 1
    if ids.shape[1] < need:
        ids = tok(TEXT * 3, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] < need:
        raise RuntimeError(f"need {need} tokens, got {ids.shape[1]}")
    cal = ids[:, : a.cal_tokens]
    ev = ids[:, a.cal_tokens : need]
    eval_input = ev[:, :-1]
    groups = contiguous_groups(len(layers), a.group_size)

    # Separate source representations. Never use FP32 decomposition tensors in the
    # deployment-matched row-Q4 candidate.
    base_nll = nll(base, ev)
    rowq4 = copy.deepcopy(base).eval()
    project_model_row_q4_(rowq4)
    rowq4_nll = nll(rowq4, ev)
    baseline_bytes = modeled_bytes(rowq4)

    fitted_fp32, sweep_fp32 = fit_all(base, cal, eval_input, groups, ranks)
    fitted_q4, sweep_q4 = fit_all(rowq4, cal, eval_input, groups, ranks)
    schedules = schedule_from_q4(fitted_q4)

    e2e = {}
    for name, sch in schedules.items():
        # FP32 diagnostic at deployment-selected ranks, but using FP32-fitted factors.
        m = copy.deepcopy(base).eval()
        _, selected = replace_with_schedule(m, fitted_fp32, sch, False)
        struct_nll = nll(m, ev)

        # Deployment-matched path: row-Q4 source -> row-Q4 activation fit -> group64 factors.
        mq = copy.deepcopy(rowq4).eval()
        qids, qselected = replace_with_schedule(mq, fitted_q4, sch, True)
        qnll = nll(mq, ev)
        qbytes = modeled_bytes(mq, qids)
        e2e[name] = {
            "gate": GATES[name],
            "rank_selection_source": "row-Q4 held-out post-Q4_GROUP64 factor operator error",
            "selected_groups": qselected,
            "selected_group_count": len(qselected),
            "fp32_factor_nll": struct_nll,
            "delta_nats_vs_fp32": struct_nll - base_nll,
            "ppl_ratio_vs_fp32": math.exp(struct_nll - base_nll),
            "q4_factor_nll": qnll,
            "delta_nats_vs_row_q4": qnll - rowq4_nll,
            "ppl_ratio_vs_row_q4": math.exp(qnll - rowq4_nll),
            "modeled_rowq4_baseline_bytes": baseline_bytes,
            "modeled_candidate_bytes": qbytes,
            "whole_model_weight_reduction_x": baseline_bytes / qbytes,
        }
        del m, mq

    out = {
        "run": 7,
        "evidence_level": "L3 real pretrained segmented shared-basis experiment",
        "model": a.model,
        "model_commit": getattr(base.config, "_commit_hash", None),
        "architecture": {
            "layers": len(layers),
            "hidden_size": getattr(base.config, "hidden_size", None),
            "intermediate_size": getattr(base.config, "intermediate_size", None),
        },
        "protocol": {
            "calibration_tokens": a.cal_tokens,
            "evaluation_tokens": a.eval_tokens,
            "group_size": a.group_size,
            "ranks": ranks,
            "sites": SITES,
            "basis_fit": "shared output basis from randomized PCA of stacked calibration operator outputs; layer-unique C_i=B^T W_i",
            "representation_matching": "FP32 and row-Q4 source models are fitted independently from their own activations and weights; deployment rank selection uses only row-Q4 post-factor-quantization held-out errors",
            "factor_storage": "Q4_GROUP64 basis and coefficients in deployment candidate",
            "down_proj": "left uncompressed because Run6 identified it as the least low-rank family",
            "gates_precommitted": GATES,
        },
        "quality_baselines": {
            "fp32_nll": base_nll,
            "row_q4_nll": rowq4_nll,
            "row_q4_ppl_ratio_vs_fp32": math.exp(rowq4_nll - base_nll),
        },
        "operator_sweep_fp32": sweep_fp32,
        "operator_sweep_rowq4": sweep_q4,
        "rank_schedules": {k: strip_schedule(v) for k, v in schedules.items()},
        "end_to_end": e2e,
        "claim_boundary": "Real-model structural/factor-quantization experiment. Modeled weight bytes are not RSS/VRAM. No latent-KV integration or native shared-basis kernel is claimed.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"baselines": out["quality_baselines"], "end_to_end": e2e}, indent=2))


if __name__ == "__main__":
    main()
