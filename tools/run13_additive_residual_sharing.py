#!/usr/bin/env python3
"""Run 13: clustered full-rank base + low-rank per-layer residual sharing.

Hypothesis:
    W_l ~= B_group + U_l V_l^T

This is distinct from:
- Run 6 exact/near-exact whole-block sharing: every logical layer retains a
  layer-specific residual here;
- Run 7 shared low-rank output bases: the shared component here remains full-rank,
  and only inter-layer differences are constrained to low rank.

The harness tests all 30 SmolLM2 decoder layers for q/k/v/o/gate/up/down. Bases
and adapter factors are also projected to the project's Q4_GROUP64 arithmetic so
quality and byte accounting use the same representation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 30
GROUP = 64


def q4_group64(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Signed [-8,7] group64 Q4 with one FP16 absmax/7 scale/group.

    Byte contract matches the project's Q4_GROUP64 convention: packed nibbles
    plus one FP16 scale for every contiguous <=64-weight row group.
    """
    w = w.float().contiguous()
    rows, cols = w.shape
    out = torch.empty_like(w)
    total_bytes = 0
    for s in range(0, cols, GROUP):
        e = min(cols, s + GROUP)
        x = w[:, s:e]
        scale = x.abs().amax(dim=1, keepdim=True) / 7.0
        scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
        qi = torch.round(x / scale).clamp_(-8, 7)
        out[:, s:e] = qi * scale
        width = e - s
        total_bytes += rows * (math.ceil(width / 2) + 2)
    return out, total_bytes


def q4_bytes(rows: int, cols: int) -> int:
    total = 0
    for s in range(0, cols, GROUP):
        width = min(GROUP, cols - s)
        total += rows * (math.ceil(width / 2) + 2)
    return total


def groups_for(n_bases: int) -> list[list[int]]:
    if N_LAYERS % n_bases:
        raise ValueError(f"{n_bases=} must divide {N_LAYERS}")
    width = N_LAYERS // n_bases
    return [list(range(i * width, (i + 1) * width)) for i in range(n_bases)]


def resolve_modules(model) -> dict[str, torch.nn.Module]:
    result = {}
    for li, layer in enumerate(model.model.layers):
        for site in SITES:
            parent = layer.self_attn if site in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp
            result[f"layer{li}.{site}"] = getattr(parent, site)
    return result


def collect_inputs(
    model,
    tokenizer,
    modules: dict[str, torch.nn.Module],
    corpus: Path,
    rows: int,
    windows: int,
    ctx: int,
) -> dict[str, torch.Tensor]:
    text = corpus.read_text(errors="replace")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < windows * ctx:
        raise RuntimeError(f"Need at least {windows*ctx} tokens in {corpus}; found {ids.numel()}")
    stores: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    counts = {name: 0 for name in modules}
    take_per_window = math.ceil(rows / windows)
    handles = []

    def hook_for(name: str):
        def hook(_module, inp, _out):
            if counts[name] >= rows:
                return
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            take = min(take_per_window, rows - counts[name], x.shape[0])
            if take <= 0:
                return
            if take < x.shape[0]:
                pos = torch.linspace(0, x.shape[0] - 1, steps=take).round().long()
                x = x.index_select(0, pos)
            stores[name].append(x[:take].cpu())
            counts[name] += take
        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(hook_for(name)))
    try:
        with torch.inference_mode():
            for wi in range(windows):
                seq = ids[wi * ctx : (wi + 1) * ctx].unsqueeze(0)
                model(input_ids=seq, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    result = {}
    for name, parts in stores.items():
        if not parts:
            raise RuntimeError(f"No activations for {name}")
        result[name] = torch.cat(parts, dim=0)[:rows].contiguous()
    return result


def ridge_pinv(x: torch.Tensor, damp: float = 1e-3) -> torch.Tensor:
    """Return X^T (X X^T + lambda I)^-1 using a sample-space solve."""
    gram = x @ x.T
    diag = float(torch.diag(gram).mean())
    lam = max(1e-8, damp * diag)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    try:
        inv = torch.linalg.solve(gram + lam * eye, eye)
    except RuntimeError:
        inv = torch.linalg.pinv(gram + lam * eye)
    return x.T @ inv


def fit_activation_residual_factors(
    delta: torch.Tensor,
    x: torch.Tensor,
    pinv: torch.Tensor,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit rank<=max_rank mapping to calibration residual outputs.

    Y_delta = X delta^T. SVD gives its best rank-r calibration output subspace.
    With P = regularized pinv(X), A^T = P U S Vh. We store V_in = P U S and
    U_out = Vh^T, so X V_in U_out^T approximates X delta^T.
    """
    y = x @ delta.T
    u, s, vh = torch.linalg.svd(y, full_matrices=False)
    r = min(max_rank, s.numel())
    us = u[:, :r] * s[:r]
    v_in = pinv @ us                  # [in, r]
    u_out = vh[:r, :].T.contiguous() # [out, r]
    return u_out, v_in


def nmse_parts(y: torch.Tensor, yh: torch.Tensor) -> tuple[float, float, float]:
    num = float((yh - y).square().sum())
    den = float(y.square().sum())
    return num / max(den, 1e-30), num, den


def aggregate(rows: list[dict[str, Any]], compressed_bytes: int, baseline_bytes: int) -> dict[str, Any]:
    vals = [r["heldout_output_nmse"] for r in rows]
    num = sum(r["nmse_numerator"] for r in rows)
    den = sum(r["nmse_denominator"] for r in rows)
    by_family = {}
    for site in SITES:
        sv = [r["heldout_output_nmse"] for r in rows if r["operator"] == site]
        by_family[site] = {
            "median_nmse": statistics.median(sv),
            "mean_nmse": statistics.fmean(sv),
            "fraction_lt_0_05": sum(v < 0.05 for v in sv) / len(sv),
        }
    return {
        "site_count": len(rows),
        "median_site_nmse": statistics.median(vals),
        "mean_site_nmse": statistics.fmean(vals),
        "energy_weighted_global_nmse": num / max(den, 1e-30),
        "fraction_site_nmse_lt_0_05": sum(v < 0.05 for v in vals) / len(vals),
        "fraction_site_nmse_lt_0_10": sum(v < 0.10 for v in vals) / len(vals),
        "baseline_independent_q4_group64_bytes": baseline_bytes,
        "compressed_q4_base_plus_adapter_bytes": compressed_bytes,
        "main_projection_pool_reduction_x": baseline_bytes / compressed_bytes,
        "by_operator": by_family,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--cal-rows", type=int, default=128)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n-bases", default="2,3,6")
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.calibration.resolve() == args.evaluation.resolve():
        raise RuntimeError("Calibration and evaluation files must be disjoint")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    n_bases_values = tuple(int(x) for x in args.n_bases.split(","))
    ranks = tuple(sorted(int(x) for x in args.ranks.split(",")))
    max_rank = max(ranks)

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    modules = resolve_modules(model)

    cal = collect_inputs(model, tok, modules, args.calibration, args.cal_rows, args.windows, args.ctx)
    ev = collect_inputs(model, tok, modules, args.evaluation, args.eval_rows, args.windows, args.ctx)
    pinvs = {name: ridge_pinv(x) for name, x in cal.items()}

    weights = {name: mod.weight.detach().float().cpu().contiguous() for name, mod in modules.items()}
    baseline_bytes = sum(q4_bytes(*w.shape) for w in weights.values())

    all_configs = []
    with torch.inference_mode():
        for n_bases in n_bases_values:
            layer_groups = groups_for(n_bases)
            base_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, int]] = {}
            # Fit and quantize one full-rank base per operator/group.
            for site in SITES:
                for gi, layers in enumerate(layer_groups):
                    stack = torch.stack([weights[f"layer{li}.{site}"] for li in layers], dim=0)
                    base = stack.mean(dim=0)
                    base_q, bbytes = q4_group64(base)
                    base_cache[(site, gi)] = (base, base_q, bbytes)

            # Per-layer factor fits at max rank are reusable for all requested truncations.
            factor_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            layer_to_group = {li: gi for gi, layers in enumerate(layer_groups) for li in layers}
            for li in range(N_LAYERS):
                for site in SITES:
                    name = f"layer{li}.{site}"
                    gi = layer_to_group[li]
                    base = base_cache[(site, gi)][0]
                    delta = weights[name] - base
                    u_out, v_in = fit_activation_residual_factors(delta, cal[name], pinvs[name], max_rank)
                    factor_cache[(site, li)] = (delta, u_out, v_in)

            for rank in ranks:
                rows = []
                compressed_bytes = sum(item[2] for item in base_cache.values())
                # Adapter bytes are charged separately for every logical layer.
                adapter_bytes = 0
                unquantized_rows = []
                for li in range(N_LAYERS):
                    for site in SITES:
                        name = f"layer{li}.{site}"
                        gi = layer_to_group[li]
                        base, base_q, _ = base_cache[(site, gi)]
                        _delta, u_max, v_max = factor_cache[(site, li)]
                        r = min(rank, u_max.shape[1])
                        u = u_max[:, :r].contiguous()
                        v = v_max[:, :r].contiguous()

                        # Exact-factor structural ceiling.
                        y = ev[name] @ weights[name].T
                        yh_fp = ev[name] @ base.T + (ev[name] @ v) @ u.T
                        fp_nmse, fp_num, fp_den = nmse_parts(y, yh_fp)
                        unquantized_rows.append({
                            "layer": li,
                            "operator": site,
                            "heldout_output_nmse": fp_nmse,
                            "nmse_numerator": fp_num,
                            "nmse_denominator": fp_den,
                        })

                        # Deployment-representation diagnostic: Q4 base + Q4 factors.
                        uq, ubytes = q4_group64(u)
                        vtq, vbytes = q4_group64(v.T.contiguous())
                        adapter_bytes += ubytes + vbytes
                        yh = ev[name] @ base_q.T + (ev[name] @ vtq.T) @ uq.T
                        q_nmse, q_num, q_den = nmse_parts(y, yh)
                        rows.append({
                            "layer": li,
                            "operator": site,
                            "group_index": gi,
                            "rank": r,
                            "heldout_output_nmse": q_nmse,
                            "nmse_numerator": q_num,
                            "nmse_denominator": q_den,
                            "adapter_q4_bytes": ubytes + vbytes,
                            "weight_shape": list(weights[name].shape),
                        })

                compressed_total = compressed_bytes + adapter_bytes
                packed = aggregate(rows, compressed_total, baseline_bytes)
                # Structural ceiling uses parameter-equivalent FP16 base/factors only for quality;
                # byte reduction is reported separately below so it cannot be confused with packed Q4.
                ceiling_vals = [r["heldout_output_nmse"] for r in unquantized_rows]
                ceiling_num = sum(r["nmse_numerator"] for r in unquantized_rows)
                ceiling_den = sum(r["nmse_denominator"] for r in unquantized_rows)

                # Parameter-count reduction independent of dtype/metadata.
                unique_base_params = 0
                adapter_params = 0
                independent_params = 0
                for site in SITES:
                    # all matrices in an operator family share shape
                    w0 = weights[f"layer0.{site}"]
                    out_dim, in_dim = w0.shape
                    independent_params += N_LAYERS * out_dim * in_dim
                    unique_base_params += n_bases * out_dim * in_dim
                    adapter_params += N_LAYERS * rank * (out_dim + in_dim)
                structural_params = unique_base_params + adapter_params

                config = {
                    "n_bases": n_bases,
                    "logical_layers_per_base": N_LAYERS // n_bases,
                    "adapter_rank": rank,
                    "structural_parameter_reduction_x": independent_params / structural_params,
                    "unquantized_factor_ceiling": {
                        "median_site_nmse": statistics.median(ceiling_vals),
                        "mean_site_nmse": statistics.fmean(ceiling_vals),
                        "energy_weighted_global_nmse": ceiling_num / max(ceiling_den, 1e-30),
                        "fraction_site_nmse_lt_0_05": sum(v < 0.05 for v in ceiling_vals) / len(ceiling_vals),
                    },
                    "packed_q4_group64": packed,
                    "per_site_packed": rows,
                }
                all_configs.append(config)

    # Precommitted mechanism gate: a candidate must simultaneously make a large
    # structural move and preserve held-out operator outputs. This is not an
    # end-to-end model gate.
    passing = []
    borderline = []
    for c in all_configs:
        a = c["packed_q4_group64"]
        if a["main_projection_pool_reduction_x"] >= 5.0 and a["energy_weighted_global_nmse"] <= 0.05:
            passing.append(c)
        elif a["main_projection_pool_reduction_x"] >= 4.0 and a["energy_weighted_global_nmse"] <= 0.10:
            borderline.append(c)

    def score(c: dict[str, Any]) -> tuple[float, float]:
        a = c["packed_q4_group64"]
        return (a["energy_weighted_global_nmse"], -a["main_projection_pool_reduction_x"])

    best = min(all_configs, key=score)
    decision = "pass_component_gate" if passing else ("borderline_component_gate" if borderline else "fail_component_gate")

    summary_configs = []
    for c in all_configs:
        cc = {k: v for k, v in c.items() if k != "per_site_packed"}
        summary_configs.append(cc)

    out = {
        "run": 13,
        "kind": "real_model_additive_residual_sharing_diagnostic",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "hypothesis": "W_layer ~= full_rank_shared_base_group + low_rank_layer_specific_residual",
        "distinction_from_prior_runs": {
            "run6": "logical layers are not forced to be identical; each retains a learned low-rank residual",
            "run7": "shared component remains full-rank; only inter-layer differences are low-rank",
        },
        "calibration": {
            "file": str(args.calibration),
            "rows_per_site": args.cal_rows,
            "windows": args.windows,
            "ctx": args.ctx,
        },
        "evaluation": {
            "file": str(args.evaluation),
            "rows_per_site": args.eval_rows,
            "disjoint_from_calibration": True,
        },
        "representation": {
            "shared_bases": "Q4_GROUP64",
            "adapter_U": "Q4_GROUP64",
            "adapter_V_transpose": "Q4_GROUP64",
            "no_dense_shadow_weights_counted": True,
            "baseline_for_projection_pool": "independent Q4_GROUP64 copies of all q/k/v/o/gate/up/down matrices",
        },
        "precommitted_component_gate": {
            "pass": "packed main-projection reduction >=5x AND held-out energy-weighted global operator NMSE <=0.05",
            "borderline": "packed main-projection reduction >=4x AND held-out energy-weighted global operator NMSE <=0.10",
            "note": "Operator/component gate only; end-to-end WikiText/task quality is still mandatory before L3 representation promotion.",
        },
        "configs": summary_configs,
        "decision": decision,
        "passing_config_count": len(passing),
        "borderline_config_count": len(borderline),
        "best_by_nmse_then_reduction": {k: v for k, v in best.items() if k != "per_site_packed"},
        "claim_boundary": (
            "Real pretrained held-out operator diagnostic over the main decoder projection pool only. "
            "No embeddings, norms, LM head, end-to-end perplexity, native packed runtime, RSS or VRAM claim. "
            "A component-gate pass is not a usable LARC model."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "passing": len(passing),
        "borderline": len(borderline),
        "configs": [
            {
                "n_bases": c["n_bases"],
                "layers_per_base": c["logical_layers_per_base"],
                "rank": c["adapter_rank"],
                "structural_parameter_reduction_x": c["structural_parameter_reduction_x"],
                "packed_reduction_x": c["packed_q4_group64"]["main_projection_pool_reduction_x"],
                "packed_global_nmse": c["packed_q4_group64"]["energy_weighted_global_nmse"],
                "packed_median_nmse": c["packed_q4_group64"]["median_site_nmse"],
                "fp_ceiling_global_nmse": c["unquantized_factor_ceiling"]["energy_weighted_global_nmse"],
            }
            for c in all_configs
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
