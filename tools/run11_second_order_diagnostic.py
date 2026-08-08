#!/usr/bin/env python3
"""Run 11: real-model second-order / rotation / outlier diagnostic.

This is intentionally an operator-level falsification harness, not a promoted
codec. It tests whether the dominant SmolLM2 projection matrices become
materially more 2-bit-friendly when we add information that Run 8 ignored:

* block activation covariance (GPTQ-style error feedback);
* fixed blockwise randomized Hadamard rotations;
* an arbitrary learned orthogonal block rotation as an upper-bound diagnostic;
* explicit high-salience outlier residual columns with byte accounting.

All reconstruction error is measured on held-out activations. Calibration and
evaluation corpora must be disjoint files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

SITE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_LAYERS = (0, 5, 10, 15, 20, 25, 29)
GROUP = 64


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & 0x7FFFFFFF


def hadamard(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def asym_q2_group64(w: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Four-level asymmetric Q2 with FP16 min+scale per row/group."""
    out, inn = w.shape
    q = torch.empty_like(w)
    groups = 0
    for s in range(0, inn, GROUP):
        e = min(s + GROUP, inn)
        x = w[:, s:e]
        mn = x.amin(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        scale = (mx - mn) / 3.0
        scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
        codes = torch.round((x - mn) / scale).clamp_(0, 3)
        q[:, s:e] = mn + codes * scale
        groups += 1
    # dense 2-bit codes + FP16 min and FP16 scale per row/group
    bits = out * inn * 2 + out * groups * 32
    return q, (bits + 7) // 8


def gptq_style_q2_group64(w: torch.Tensor, x_cal: torch.Tensor, damp: float = 0.01) -> tuple[torch.Tensor, int]:
    """Block-diagonal GPTQ-style sequential error feedback.

    This is not claimed to reproduce a particular external GPTQ implementation.
    Each 64-input block uses its measured activation covariance, a damped inverse
    Hessian Cholesky factor, and fixed asymmetric four-level row/group ranges.
    Runtime representation is still ordinary Q2 + min/scale metadata.
    """
    out, inn = w.shape
    q_all = torch.empty_like(w)
    groups = 0
    for s in range(0, inn, GROUP):
        e = min(s + GROUP, inn)
        wb = w[:, s:e].clone()
        xb = x_cal[:, s:e]
        width = e - s
        h = (xb.T @ xb) / max(1, xb.shape[0])
        diag_mean = float(torch.diag(h).mean()) if width else 0.0
        h = h + torch.eye(width, dtype=h.dtype) * max(1e-8, damp * diag_mean)
        try:
            hinv = torch.linalg.inv(h)
            chol = torch.linalg.cholesky(hinv, upper=True)
        except RuntimeError:
            hinv = torch.linalg.pinv(h)
            # Numerical fallback: PSD projection before Cholesky.
            ev, u = torch.linalg.eigh((hinv + hinv.T) * 0.5)
            hinv = (u * ev.clamp_min(1e-10)) @ u.T
            chol = torch.linalg.cholesky(hinv, upper=True)

        orig = wb.clone()
        mn = orig.amin(dim=1)
        mx = orig.amax(dim=1)
        scale = (mx - mn) / 3.0
        scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
        qb = torch.empty_like(wb)
        for i in range(width):
            wi = wb[:, i]
            qi = mn + torch.round((wi - mn) / scale).clamp_(0, 3) * scale
            qb[:, i] = qi
            denom = float(chol[i, i])
            if abs(denom) < 1e-12:
                continue
            err = (wi - qi) / denom
            if i + 1 < width:
                wb[:, i + 1 :] -= err[:, None] * chol[i, i + 1 :][None, :]
        q_all[:, s:e] = qb
        groups += 1
    bits = out * inn * 2 + out * groups * 32
    return q_all, (bits + 7) // 8


def random_hadamard_transform(site: str, inn: int) -> list[torch.Tensor]:
    if inn % GROUP:
        raise ValueError(f"input dim {inn} not divisible by {GROUP}")
    h = hadamard(GROUP)
    mats = []
    g = torch.Generator().manual_seed(stable_seed(site))
    for _ in range(inn // GROUP):
        signs = torch.randint(0, 2, (GROUP,), generator=g, dtype=torch.int64).float().mul_(2).sub_(1)
        # M is the row-vector post-transform. Orthogonal and cheap to represent:
        # randomized signs plus an implicit Hadamard matrix.
        mats.append(torch.diag(signs) @ h)
    return mats


def eigen_transforms(x_cal: torch.Tensor) -> list[torch.Tensor]:
    inn = x_cal.shape[1]
    if inn % GROUP:
        raise ValueError(f"input dim {inn} not divisible by {GROUP}")
    mats = []
    for s in range(0, inn, GROUP):
        xb = x_cal[:, s : s + GROUP]
        cov = (xb.T @ xb) / max(1, xb.shape[0])
        _, u = torch.linalg.eigh(cov)
        mats.append(u)
    return mats


def apply_block_transform(x: torch.Tensor, mats: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty_like(x)
    for gi, m in enumerate(mats):
        s = gi * GROUP
        out[:, s : s + GROUP] = x[:, s : s + GROUP] @ m
    return out


def apply_weight_transform(w: torch.Tensor, mats: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty_like(w)
    for gi, m in enumerate(mats):
        s = gi * GROUP
        out[:, s : s + GROUP] = w[:, s : s + GROUP] @ m
    return out


def output_nmse(w_ref: torch.Tensor, w_hat: torch.Tensor, x_eval: torch.Tensor, x_hat: torch.Tensor | None = None) -> float:
    if x_hat is None:
        x_hat = x_eval
    y = x_eval @ w_ref.T
    yh = x_hat @ w_hat.T
    den = float(y.square().sum())
    return float((yh - y).square().sum()) / max(den, 1e-30)


def outlier_escape(
    w_ref: torch.Tensor,
    q_base: torch.Tensor,
    x_cal: torch.Tensor,
    fraction: float,
    base_bytes: int,
) -> tuple[torch.Tensor, int, list[int]]:
    inn = w_ref.shape[1]
    k = max(1, int(round(inn * fraction)))
    xrms = x_cal.square().mean(dim=0).sqrt()
    wrms = w_ref.square().mean(dim=0).sqrt()
    salience = xrms * wrms
    idx = torch.topk(salience, k=k, largest=True).indices
    q = q_base.clone()
    q[:, idx] = w_ref[:, idx]
    # Conservative direct-packed contract: keep the full dense Q2 payload and add
    # FP16 residual values for escaped columns plus one uint16 column index each.
    extra = w_ref.shape[0] * k * 2 + k * 2
    return q, base_bytes + extra, sorted(int(i) for i in idx)


def repr_row(name: str, nmse: float, bytes_: int, fp16_bytes: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "variant": name,
        "heldout_output_nmse": nmse,
        "encoded_bytes": int(bytes_),
        "reduction_vs_fp16_x": fp16_bytes / bytes_,
        "effective_bpw": bytes_ * 8 / (fp16_bytes / 2),
    }
    if extra:
        row.update(extra)
    return row


def resolve_modules(model, layers: tuple[int, ...]) -> dict[str, torch.nn.Module]:
    dec = model.model.layers
    out = {}
    for li in layers:
        layer = dec[li]
        for site in SITE_NAMES:
            parent = layer.self_attn if site in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp
            out[f"layer{li}.{site}"] = getattr(parent, site)
    return out


def collect_inputs(model, tokenizer, modules: dict[str, torch.nn.Module], corpus: Path, rows: int, windows: int, ctx: int) -> dict[str, torch.Tensor]:
    text = corpus.read_text(errors="replace")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needed = windows * ctx
    if ids.numel() < needed:
        raise RuntimeError(f"{corpus} has only {ids.numel()} tokens; need {needed}")
    stores: dict[str, list[torch.Tensor]] = {k: [] for k in modules}
    counts = {k: 0 for k in modules}
    per_window = math.ceil(rows / windows)
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            if counts[name] >= rows:
                return
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            take = min(per_window, rows - counts[name], x.shape[0])
            if take <= 0:
                return
            if take == x.shape[0]:
                chosen = x
            else:
                pos = torch.linspace(0, x.shape[0] - 1, steps=take).round().long()
                chosen = x.index_select(0, pos)
            stores[name].append(chosen.cpu())
            counts[name] += chosen.shape[0]
        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            for wi in range(windows):
                seq = ids[wi * ctx : (wi + 1) * ctx].unsqueeze(0)
                model(input_ids=seq, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, parts in stores.items():
        if not parts:
            raise RuntimeError(f"No activations captured for {name}")
        out[name] = torch.cat(parts, dim=0)[:rows].contiguous()
    return out


def summarize_sites(sites: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({r["variant"] for s in sites for r in s["variants"]})
    out = {}
    for v in variants:
        vals = [r["heldout_output_nmse"] for s in sites for r in s["variants"] if r["variant"] == v]
        red = [r["reduction_vs_fp16_x"] for s in sites for r in s["variants"] if r["variant"] == v]
        if vals:
            out[v] = {
                "site_count": len(vals),
                "median_output_nmse": statistics.median(vals),
                "mean_output_nmse": statistics.fmean(vals),
                "fraction_nmse_lt_0_01": sum(x < 0.01 for x in vals) / len(vals),
                "fraction_nmse_lt_0_05": sum(x < 0.05 for x in vals) / len(vals),
                "median_reduction_vs_fp16_x": statistics.median(red),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    ).eval()
    modules = resolve_modules(model, layers)

    cal = collect_inputs(model, tokenizer, modules, args.calibration, args.rows, args.windows, args.ctx)
    ev = collect_inputs(model, tokenizer, modules, args.evaluation, args.eval_rows, args.windows, args.ctx)

    site_rows = []
    with torch.inference_mode():
        for name, mod in modules.items():
            w = mod.weight.detach().float().cpu().contiguous()
            xc = cal[name]
            xe = ev[name]
            fp16_bytes = w.numel() * 2
            variants = []

            q2, q2_bytes = asym_q2_group64(w)
            variants.append(repr_row("q2_asym_group64", output_nmse(w, q2, xe), q2_bytes, fp16_bytes))

            gq2, gq2_bytes = gptq_style_q2_group64(w, xc)
            variants.append(repr_row("block_gptq_q2", output_nmse(w, gq2, xe), gq2_bytes, fp16_bytes))

            hm = random_hadamard_transform(name, w.shape[1])
            wh = apply_weight_transform(w, hm)
            xch = apply_block_transform(xc, hm)
            xeh = apply_block_transform(xe, hm)
            hq2, hbytes = asym_q2_group64(wh)
            # one sign bit/input; Hadamard itself is implicit
            hbytes_total = hbytes + math.ceil(w.shape[1] / 8)
            variants.append(repr_row(
                "hadamard_q2", output_nmse(w, hq2, xe, xeh), hbytes_total, fp16_bytes,
                {"transform_bytes": math.ceil(w.shape[1] / 8)},
            ))
            hgq2, hgbytes = gptq_style_q2_group64(wh, xch)
            hgbytes_total = hgbytes + math.ceil(w.shape[1] / 8)
            variants.append(repr_row(
                "hadamard_block_gptq_q2", output_nmse(w, hgq2, xe, xeh), hgbytes_total, fp16_bytes,
                {"transform_bytes": math.ceil(w.shape[1] / 8)},
            ))

            em = eigen_transforms(xc)
            we = apply_weight_transform(w, em)
            xce = apply_block_transform(xc, em)
            xee = apply_block_transform(xe, em)
            eq2, ebytes = asym_q2_group64(we)
            transform_bytes = len(em) * GROUP * GROUP * 2
            variants.append(repr_row(
                "learned_orthogonal_q2_ceiling", output_nmse(w, eq2, xe, xee), ebytes + transform_bytes, fp16_bytes,
                {"transform_bytes": transform_bytes, "note": "Dense FP16 64x64 block transforms; diagnostic ceiling, not preferred runtime."},
            ))

            for frac in (0.01, 0.02, 0.05):
                oq, obytes, idx = outlier_escape(w, gq2, xc, frac, gq2_bytes)
                variants.append(repr_row(
                    f"block_gptq_q2_outlier_{int(frac*100)}pct",
                    output_nmse(w, oq, xe), obytes, fp16_bytes,
                    {"outlier_fraction_requested": frac, "outlier_columns": len(idx)},
                ))
                hoq, hobytes, hidx = outlier_escape(wh, hgq2, xch, frac, hgbytes_total)
                variants.append(repr_row(
                    f"hadamard_block_gptq_q2_outlier_{int(frac*100)}pct",
                    output_nmse(w, hoq, xe, xeh), hobytes, fp16_bytes,
                    {"outlier_fraction_requested": frac, "outlier_columns": len(hidx)},
                ))

            site_rows.append({
                "site": name,
                "shape_out_in": list(w.shape),
                "fp16_bytes": fp16_bytes,
                "calibration_rows": int(xc.shape[0]),
                "evaluation_rows": int(xe.shape[0]),
                "input_absmax_over_rms": float(xc.abs().max() / xc.square().mean().sqrt().clamp_min(1e-12)),
                "weight_absmax_over_rms": float(w.abs().max() / w.square().mean().sqrt().clamp_min(1e-12)),
                "hadamard_weight_absmax_over_rms": float(wh.abs().max() / wh.square().mean().sqrt().clamp_min(1e-12)),
                "variants": variants,
            })

    out = {
        "run": 11,
        "kind": "second_order_rotation_outlier_operator_diagnostic",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "layers": list(layers),
        "sites_per_layer": list(SITE_NAMES),
        "calibration_file": str(args.calibration),
        "evaluation_file": str(args.evaluation),
        "calibration_evaluation_disjoint": args.calibration.resolve() != args.evaluation.resolve(),
        "q2_contract": "2-bit asymmetric codes + FP16 min/scale per row/group64",
        "gptq_contract": "block-diagonal activation covariance, 1% damping, inverse-Hessian Cholesky sequential error feedback; diagnostic implementation, not external GPTQ parity claim",
        "outlier_contract": "dense Q2 retained plus FP16 exact residual columns and uint16 column indices; conservative byte accounting",
        "sites": site_rows,
        "aggregate": summarize_sites(site_rows),
        "claim_boundary": "Operator-level held-out activation diagnostic only. No end-to-end perplexity, task accuracy, native packed runtime, RSS, or VRAM claim. Learned orthogonal rotation includes dense FP16 transform bytes and is an upper-bound diagnostic, not a proposed efficient runtime transform.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
