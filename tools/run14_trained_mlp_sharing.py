#!/usr/bin/env python3
"""Run 14: train nonlinear shared MLPs with tiny depth-specific FiLM state.

Hypothesis
----------
Runs 6/7/13 impose algebraic post-hoc relations between independently trained
matrices. Run 14 instead asks whether the dominant *nonlinear MLP function* can
be explicitly trained to serve several logical depths:

    x_l' = s_in[l] * x_l
    h_l  = silu(G x_l') * (U x_l')
    h_l' = s_mid[l] * h_l
    y_l  = s_out[l] * D h_l'

G/U/D are one physical full-rank MLP shared by a contiguous depth group. Only
three small FP16 scale vectors remain layer-specific. Shared matrices are trained
against cached teacher MLP input/output pairs from a real pretrained SmolLM2.

The component gate is evaluated twice:
  * FP32 trained ceiling;
  * representation-matched Q4_GROUP64 shared weights + FP16 FiLM scales.

No full-model decoder replacement, perplexity, RSS or VRAM claim is made here.
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

N_LAYERS = 30
GROUP = 64


def q4_group64(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Project a matrix to the project's signed Q4_GROUP64 representation."""
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


def collect_mlp_pairs(
    model,
    tokenizer,
    corpus: Path,
    rows: int,
    windows: int,
    ctx: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    text = corpus.read_text(errors="replace")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < windows * ctx:
        raise RuntimeError(f"{corpus} has {ids.numel()} tokens; need {windows*ctx}")

    xs: dict[int, list[torch.Tensor]] = {i: [] for i in range(N_LAYERS)}
    ys: dict[int, list[torch.Tensor]] = {i: [] for i in range(N_LAYERS)}
    counts = {i: 0 for i in range(N_LAYERS)}
    per_window = math.ceil(rows / windows)
    handles = []

    def make_hook(li: int):
        def hook(_module, inputs, output):
            if counts[li] >= rows:
                return
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            y = output.detach().float().reshape(-1, output.shape[-1])
            take = min(per_window, rows - counts[li], x.shape[0])
            if take <= 0:
                return
            if take < x.shape[0]:
                pos = torch.linspace(0, x.shape[0] - 1, steps=take).round().long()
                x = x.index_select(0, pos)
                y = y.index_select(0, pos)
            xs[li].append(x[:take].cpu())
            ys[li].append(y[:take].cpu())
            counts[li] += take
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(li)))
    try:
        with torch.inference_mode():
            for wi in range(windows):
                seq = ids[wi * ctx:(wi + 1) * ctx].unsqueeze(0)
                model(input_ids=seq, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out_x: dict[int, torch.Tensor] = {}
    out_y: dict[int, torch.Tensor] = {}
    for li in range(N_LAYERS):
        if not xs[li]:
            raise RuntimeError(f"No MLP activations captured for layer {li}")
        out_x[li] = torch.cat(xs[li], dim=0)[:rows].contiguous()
        out_y[li] = torch.cat(ys[li], dim=0)[:rows].contiguous()
    return out_x, out_y


class SharedMLP(torch.nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, logical_count: int):
        super().__init__()
        self.gate = torch.nn.Parameter(gate.clone().float())
        self.up = torch.nn.Parameter(up.clone().float())
        self.down = torch.nn.Parameter(down.clone().float())
        hidden = gate.shape[1]
        inter = gate.shape[0]
        self.scale_in = torch.nn.Parameter(torch.ones(logical_count, hidden))
        self.scale_mid = torch.nn.Parameter(torch.ones(logical_count, inter))
        self.scale_out = torch.nn.Parameter(torch.ones(logical_count, hidden))

    def forward_group(self, x: torch.Tensor) -> torch.Tensor:
        """x: [logical, batch, hidden] -> [logical, batch, hidden]."""
        g, b, d = x.shape
        xm = x * self.scale_in[:, None, :]
        flat = xm.reshape(g * b, d)
        gate = F.silu(F.linear(flat, self.gate)).reshape(g, b, -1)
        up = F.linear(flat, self.up).reshape(g, b, -1)
        h = gate * up * self.scale_mid[:, None, :]
        y = F.linear(h.reshape(g * b, -1), self.down).reshape(g, b, d)
        return y * self.scale_out[:, None, :]


def forward_packed(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale_in: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_out: torch.Tensor,
) -> torch.Tensor:
    g, b, d = x.shape
    xm = x * scale_in[:, None, :]
    flat = xm.reshape(g * b, d)
    h = F.silu(F.linear(flat, gate)).reshape(g, b, -1)
    h = h * F.linear(flat, up).reshape(g, b, -1)
    h = h * scale_mid[:, None, :]
    y = F.linear(h.reshape(g * b, -1), down).reshape(g, b, d)
    return y * scale_out[:, None, :]


def normalized_site_nmse(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    # pred/target: [logical, rows, hidden]
    num = (pred - target).square().sum(dim=(1, 2))
    den = target.square().sum(dim=(1, 2)).clamp_min(1e-30)
    return [float(v) for v in (num / den)]


def percentile(vals: list[float], q: float) -> float:
    t = torch.tensor(vals, dtype=torch.float64)
    return float(torch.quantile(t, q))


def summarize(vals: list[float]) -> dict[str, float]:
    return {
        "site_count": len(vals),
        "median_site_nmse": statistics.median(vals),
        "mean_site_nmse": statistics.fmean(vals),
        "p90_site_nmse": percentile(vals, 0.90),
        "max_site_nmse": max(vals),
        "fraction_lt_0_01": sum(v < 0.01 for v in vals) / len(vals),
        "fraction_lt_0_05": sum(v < 0.05 for v in vals) / len(vals),
        "fraction_lt_0_10": sum(v < 0.10 for v in vals) / len(vals),
    }


def group_layers(layers_per_physical: int) -> list[list[int]]:
    if N_LAYERS % layers_per_physical:
        raise ValueError("layers-per-physical must divide 30")
    return [
        list(range(start, start + layers_per_physical))
        for start in range(0, N_LAYERS, layers_per_physical)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--layers-per-physical", type=int, choices=(5, 10), required=True)
    ap.add_argument("--cal-rows", type=int, default=128)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-per-layer", type=int, default=4)
    ap.add_argument("--weight-lr", type=float, default=3e-4)
    ap.add_argument("--film-lr", type=float, default=3e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.calibration.resolve() == args.evaluation.resolve():
        raise RuntimeError("Calibration and evaluation corpora must be disjoint")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    cal_x, cal_y = collect_mlp_pairs(
        teacher, tok, args.calibration, args.cal_rows, args.windows, args.ctx
    )
    ev_x, ev_y = collect_mlp_pairs(
        teacher, tok, args.evaluation, args.eval_rows, args.windows, args.ctx
    )

    groups = group_layers(args.layers_per_physical)
    baseline_q4_bytes = 0
    for layer in teacher.model.layers:
        mlp = layer.mlp
        baseline_q4_bytes += q4_bytes(*mlp.gate_proj.weight.shape)
        baseline_q4_bytes += q4_bytes(*mlp.up_proj.weight.shape)
        baseline_q4_bytes += q4_bytes(*mlp.down_proj.weight.shape)

    group_results: list[dict[str, Any]] = []
    initial_all: list[float] = []
    fp_all: list[float] = []
    packed_all: list[float] = []
    candidate_weight_bytes = 0
    candidate_film_bytes = 0
    t_start = time.perf_counter()

    for gi, layers in enumerate(groups):
        teacher_mlps = [teacher.model.layers[li].mlp for li in layers]
        gate_mean = torch.stack([m.gate_proj.weight.detach().float() for m in teacher_mlps]).mean(0)
        up_mean = torch.stack([m.up_proj.weight.detach().float() for m in teacher_mlps]).mean(0)
        down_mean = torch.stack([m.down_proj.weight.detach().float() for m in teacher_mlps]).mean(0)
        student = SharedMLP(gate_mean, up_mean, down_mean, len(layers))

        eval_x = torch.stack([ev_x[li] for li in layers], dim=0)
        eval_y = torch.stack([ev_y[li] for li in layers], dim=0)
        with torch.inference_mode():
            init_pred = student.forward_group(eval_x)
            initial_nmse = normalized_site_nmse(init_pred, eval_y)
        initial_all.extend(initial_nmse)

        weight_params = [student.gate, student.up, student.down]
        film_params = [student.scale_in, student.scale_mid, student.scale_out]
        opt = torch.optim.AdamW(
            [
                {"params": weight_params, "lr": args.weight_lr, "weight_decay": 0.0},
                {"params": film_params, "lr": args.film_lr, "weight_decay": 0.0},
            ]
        )
        gen = torch.Generator().manual_seed(1000 + gi)
        losses = []
        for step in range(args.steps):
            batch_x = []
            batch_y = []
            for li in layers:
                n = cal_x[li].shape[0]
                idx = torch.randint(0, n, (args.batch_per_layer,), generator=gen)
                batch_x.append(cal_x[li].index_select(0, idx))
                batch_y.append(cal_y[li].index_select(0, idx))
            xb = torch.stack(batch_x, dim=0)
            yb = torch.stack(batch_y, dim=0)
            pred = student.forward_group(xb)
            per_layer_num = (pred - yb).square().mean(dim=(1, 2))
            per_layer_den = yb.square().mean(dim=(1, 2)).clamp_min(1e-12)
            loss = (per_layer_num / per_layer_den).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))

        with torch.inference_mode():
            fp_pred = student.forward_group(eval_x)
            fp_nmse = normalized_site_nmse(fp_pred, eval_y)
        fp_all.extend(fp_nmse)

        # Representation-matched deployment projection.
        gate_q, gate_bytes = q4_group64(student.gate)
        up_q, up_bytes = q4_group64(student.up)
        down_q, down_bytes = q4_group64(student.down)
        sin = student.scale_in.detach().half().float()
        smid = student.scale_mid.detach().half().float()
        sout = student.scale_out.detach().half().float()
        film_bytes = (sin.numel() + smid.numel() + sout.numel()) * 2
        with torch.inference_mode():
            packed_pred = forward_packed(eval_x, gate_q, up_q, down_q, sin, smid, sout)
            packed_nmse = normalized_site_nmse(packed_pred, eval_y)
        packed_all.extend(packed_nmse)
        candidate_weight_bytes += gate_bytes + up_bytes + down_bytes
        candidate_film_bytes += film_bytes

        group_results.append({
            "group_index": gi,
            "logical_layers": layers,
            "initial_mean_mlp": summarize(initial_nmse),
            "trained_fp32": summarize(fp_nmse),
            "packed_q4_fp16_film": summarize(packed_nmse),
            "training": {
                "steps": args.steps,
                "batch_per_layer": args.batch_per_layer,
                "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "best_loss": min(losses) if losses else None,
            },
            "packed_bytes": {
                "shared_gate_q4": gate_bytes,
                "shared_up_q4": up_bytes,
                "shared_down_q4": down_bytes,
                "layer_specific_fp16_film": film_bytes,
            },
            "film_statistics": {
                "scale_in_abs_deviation_from_one_mean": float((sin - 1).abs().mean()),
                "scale_mid_abs_deviation_from_one_mean": float((smid - 1).abs().mean()),
                "scale_out_abs_deviation_from_one_mean": float((sout - 1).abs().mean()),
            },
        })

    candidate_bytes = candidate_weight_bytes + candidate_film_bytes
    reduction = baseline_q4_bytes / candidate_bytes
    initial_summary = summarize(initial_all)
    fp_summary = summarize(fp_all)
    packed_summary = summarize(packed_all)
    worst_group_median = max(g["packed_q4_fp16_film"]["median_site_nmse"] for g in group_results)

    # Site-normalized gate committed before result. Unlike Run 13, no raw output
    # energy is allowed to weight one logical layer over another.
    passes = (
        reduction >= 8.0
        and packed_summary["median_site_nmse"] <= 0.05
        and packed_summary["p90_site_nmse"] <= 0.15
        and worst_group_median <= 0.10
    )
    borderline = (
        reduction >= 7.0
        and packed_summary["median_site_nmse"] <= 0.10
        and packed_summary["p90_site_nmse"] <= 0.25
        and worst_group_median <= 0.15
    )
    decision = "pass_component_gate" if passes else ("borderline_component_gate" if borderline else "fail_component_gate")

    out = {
        "run": 14,
        "kind": "trained_nonlinear_shared_mlp_function_diagnostic",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "hypothesis": "one trained full-rank nonlinear MLP per depth group plus tiny depth-specific FiLM scales can replace independently trained MLPs",
        "scope": {
            "logical_layers": N_LAYERS,
            "layers_per_physical_mlp": args.layers_per_physical,
            "physical_mlp_count": len(groups),
            "targeted_projection_pool": "gate_proj + up_proj + down_proj",
            "targeted_main_projection_fraction": 0.75,
        },
        "data": {
            "calibration_file": str(args.calibration),
            "evaluation_file": str(args.evaluation),
            "disjoint": True,
            "calibration_rows_per_layer": args.cal_rows,
            "evaluation_rows_per_layer": args.eval_rows,
            "context": args.ctx,
        },
        "training": {
            "teacher_pairs": "cached real pretrained MLP input/output activations",
            "loss": "mean of per-logical-layer normalized MLP-output MSE",
            "steps_per_physical_group": args.steps,
            "batch_per_logical_layer": args.batch_per_layer,
            "weight_lr": args.weight_lr,
            "film_lr": args.film_lr,
            "shared_matrix_initialization": "arithmetic mean of the group's teacher MLP matrices",
        },
        "representation": {
            "physical_gate_up_down": "Q4_GROUP64 after function-space training",
            "logical_depth_conditioning": "FP16 elementwise scale_in[576] + scale_mid[1536] + scale_out[576] per logical layer",
            "dense_shadow_mlps_counted": False,
            "baseline": "independent Q4_GROUP64 gate/up/down for all 30 teacher MLPs",
        },
        "bytes": {
            "baseline_independent_mlp_q4_bytes": baseline_q4_bytes,
            "candidate_shared_q4_weight_bytes": candidate_weight_bytes,
            "candidate_fp16_film_bytes": candidate_film_bytes,
            "candidate_total_bytes": candidate_bytes,
            "mlp_pool_reduction_x": reduction,
        },
        "quality": {
            "untrained_group_mean_with_unit_film": initial_summary,
            "trained_fp32_ceiling": fp_summary,
            "packed_q4_fp16_film": packed_summary,
            "worst_packed_group_median_nmse": worst_group_median,
        },
        "groups": group_results,
        "precommitted_component_gate": {
            "pass": "MLP-pool reduction >=8x; packed median site NMSE <=0.05; packed p90 <=0.15; worst physical-group median <=0.10",
            "borderline": "MLP-pool reduction >=7x; packed median <=0.10; packed p90 <=0.25; worst physical-group median <=0.15",
            "aggregation": "site-normalized per-logical-layer MLP-output NMSE only; no raw cross-site output-energy weighting",
        },
        "decision": decision,
        "wall_seconds": time.perf_counter() - t_start,
        "claim_boundary": (
            "Real-pretrained nonlinear MLP component diagnostic only. The 75% figure refers to the seven main decoder "
            "projection matrices, not total model bytes. No attention replacement, full-model perplexity, task quality, "
            "native fused runtime, RSS, or VRAM claim. A component-gate pass requires subsequent end-to-end integration."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "layers_per_physical": args.layers_per_physical,
        "decision": decision,
        "bytes": out["bytes"],
        "initial": initial_summary,
        "fp32": fp_summary,
        "packed": packed_summary,
        "worst_group_median": worst_group_median,
    }, indent=2))


if __name__ == "__main__":
    main()
