#!/usr/bin/env python3
"""Run 12: derive same-parameter compression feasibility from Run-9 measurements.

This script deliberately separates three questions:

1. FILE BOUND (exact): given the actual Q4_K_M GGUF bytes and exact parameter
   count, what effective bits/parameter would a 10x/20x/30x smaller file require?
2. SAME-PARAMETER CODEC BOUND (exact arithmetic): even with zero metadata, how
   far can fixed 2/1/0.5-bit payloads move the Q4_K_M file-size ratio?
3. TOTAL-RSS ILLUSTRATION (explicit model, not a measurement): if Q4 file bytes
   correspond one-for-one to removable resident weight bytes, what fixed RSS
   floor remains? This is intentionally labeled as an optimistic residency model
   and never promoted as a measured LARC memory result.

The point is to prevent a physically impossible target from being assigned to a
codec. If 10x vs Q4 requires substantially below 1 bit per original parameter,
then ordinary same-parameter quantization cannot reach it; parameter-count
reduction, structural reuse, entropy/dictionary structure, or learned architecture
changes are mathematically required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TARGETS = (10, 20, 30)
CANDIDATE_BPW = (4.0, 3.0, 2.5, 2.0, 1.58, 1.0, 0.5, 0.25)

# SmolLM2-135M main projection geometry, checked against HF config in Run 6.
D_MODEL = 576
D_KV = 192       # 3 KV heads * 64
D_FF = 1536
N_LAYERS = 30


def projection_geometry() -> dict[str, Any]:
    per_layer = {
        "q_proj": D_MODEL * D_MODEL,
        "k_proj": D_KV * D_MODEL,
        "v_proj": D_KV * D_MODEL,
        "o_proj": D_MODEL * D_MODEL,
        "gate_proj": D_FF * D_MODEL,
        "up_proj": D_FF * D_MODEL,
        "down_proj": D_MODEL * D_FF,
    }
    total = sum(per_layer.values())
    return {
        "per_layer_parameters": per_layer,
        "per_layer_total": total,
        "all_30_layers_total": total * N_LAYERS,
        "fractions_of_main_projection_pool": {k: v / total for k, v in per_layer.items()},
        "q_plus_k_fraction": (per_layer["q_proj"] + per_layer["k_proj"]) / total,
        "q_k_gate_fraction": (per_layer["q_proj"] + per_layer["k_proj"] + per_layer["gate_proj"]) / total,
        "mlp_gate_up_down_fraction": (
            per_layer["gate_proj"] + per_layer["up_proj"] + per_layer["down_proj"]
        ) / total,
    }


def first_bench_param_count(run9: dict[str, Any]) -> int:
    for quant in ("Q4_K_M", "Q2_K"):
        q = run9.get("quantized", {}).get(quant, {})
        for category in ("prompt_processing", "generation"):
            for row in q.get("throughput", {}).get(category, []):
                n = int(row.get("model_n_params", 0) or 0)
                if n > 0:
                    return n
    raise RuntimeError("Run-9 artifact does not contain a positive llama-bench model_n_params")


def rss_table(run9: dict[str, Any], q4_file_bytes: int, q4_effective_bpw: float) -> dict[str, Any]:
    q4 = run9["quantized"]["Q4_K_M"]
    memory = q4.get("maxrss_and_reported_buffers") or q4.get("maxrss")
    if not memory:
        raise RuntimeError("Run-9 artifact has no Q4_K_M MaxRSS table")
    out: dict[str, Any] = {}
    for ctx, modes in memory.items():
        out[ctx] = {}
        for mode, row in modes.items():
            rss = int(row["median_maxrss_bytes"])
            # Explicit illustrative model: serialized Q4 bytes are assumed to be
            # exactly the removable resident-weight pool. This is intentionally
            # simplistic and is NOT claimed as measured decomposition.
            fixed = max(0, rss - q4_file_bytes)
            candidates = {}
            for bpw in CANDIDATE_BPW:
                scaled_weight = q4_file_bytes * (bpw / q4_effective_bpw)
                modeled_total = fixed + scaled_weight
                candidates[str(bpw)] = {
                    "modeled_weight_bytes": scaled_weight,
                    "modeled_total_rss_bytes": modeled_total,
                    "reduction_vs_q4_measured_rss_x": rss / modeled_total if modeled_total else math.inf,
                }
            zero_weight_max = rss / fixed if fixed > 0 else math.inf
            out[ctx][mode] = {
                "measured_q4_median_maxrss_bytes": rss,
                "illustrative_fixed_floor_bytes": fixed,
                "illustrative_zero_weight_max_reduction_x": zero_weight_max,
                "candidate_payloads": candidates,
                "warning": (
                    "Illustrative only: assumes Q4 serialized file bytes equal the entire removable resident-weight pool. "
                    "llama.cpp repacking, page residency, allocator behavior and kernel-specific scratch can violate this mapping."
                ),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run9", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run9 = json.loads(args.run9.read_text())
    if run9.get("baseline") != "llama.cpp Q4_K_M":
        raise RuntimeError("Expected a Run-9 artifact whose named baseline is llama.cpp Q4_K_M")

    params = first_bench_param_count(run9)
    q4_bytes = int(run9["quantized"]["Q4_K_M"]["file"]["bytes"])
    q2_bytes = int(run9["quantized"]["Q2_K"]["file"]["bytes"])
    f16_bytes = int(run9["f16_reference"]["file"]["bytes"])
    q4_eff_bpw = q4_bytes * 8 / params
    q2_eff_bpw = q2_bytes * 8 / params
    f16_eff_bpw = f16_bytes * 8 / params

    exact_file_targets = {}
    for target in TARGETS:
        budget = q4_bytes / target
        required_bpw = budget * 8 / params
        exact_file_targets[str(target)] = {
            "max_total_file_bytes": budget,
            "required_effective_total_file_bpw": required_bpw,
            "below_one_bit_per_original_parameter": required_bpw < 1.0,
            "below_half_bit_per_original_parameter": required_bpw < 0.5,
        }

    same_param = {}
    for bpw in CANDIDATE_BPW:
        ideal_payload = params * bpw / 8
        same_param[str(bpw)] = {
            "ideal_payload_bytes_zero_metadata": ideal_payload,
            "max_file_reduction_vs_q4_x_if_no_other_bytes": q4_bytes / ideal_payload,
            "can_reach_10x_file_ratio": ideal_payload <= q4_bytes / 10,
            "can_reach_20x_file_ratio": ideal_payload <= q4_bytes / 20,
            "can_reach_30x_file_ratio": ideal_payload <= q4_bytes / 30,
        }

    geom = projection_geometry()
    # Arithmetic consequence of Run-11's Q/K-positive observation.
    qk_frac = geom["q_plus_k_fraction"]
    mixed_projection_examples = {
        "qk_2_5_rest_4_bpw": qk_frac * 2.5 + (1 - qk_frac) * 4.0,
        "qk_2_rest_4_bpw": qk_frac * 2.0 + (1 - qk_frac) * 4.0,
        "qk_gate_2_5_rest_4_bpw": geom["q_k_gate_fraction"] * 2.5 + (1 - geom["q_k_gate_fraction"]) * 4.0,
    }

    out = {
        "run": 12,
        "kind": "compression_feasibility_bound",
        "source_run9": str(args.run9),
        "source_model": run9.get("source_model"),
        "source_model_commit": run9.get("source_model_commit"),
        "exact_baseline": {
            "parameter_count_from_llama_bench": params,
            "f16_file_bytes": f16_bytes,
            "q4_k_m_file_bytes": q4_bytes,
            "q2_k_file_bytes": q2_bytes,
            "f16_effective_file_bpw": f16_eff_bpw,
            "q4_k_m_effective_file_bpw": q4_eff_bpw,
            "q2_k_effective_file_bpw": q2_eff_bpw,
        },
        "exact_q4_relative_file_targets": exact_file_targets,
        "ideal_same_parameter_zero_metadata_bounds": same_param,
        "smollm2_main_projection_geometry": geom,
        "run11_operator_adaptive_arithmetic": {
            "mixed_projection_average_bpw_examples": mixed_projection_examples,
            "interpretation": (
                "Q+K are only 12.5% of the seven main projection matrices. Aggressively compressing Q/K alone cannot "
                "produce extreme whole-model reduction; the 75% MLP gate/up/down pool dominates the projection budget."
            ),
        },
        "illustrative_total_rss_model": rss_table(run9, q4_bytes, q4_eff_bpw),
        "decision_rule": {
            "same_parameter_codec_only": (
                "If the exact 10x Q4 file budget is below 1 effective bit per original parameter, ordinary fixed-bit "
                "same-parameter PTQ cannot reach the target even before metadata/runtime overhead. The project must "
                "obtain additional gains from parameter-count reduction, structural reuse, entropy/dictionary structure "
                "with genuinely sub-bit average information, or a model trained for compressibility."
            ),
            "total_memory": (
                "Use measured Run-9 allocator/RSS data to set hardware/runtime floors. Do not infer total-RSS feasibility "
                "from file bpw alone; the illustrative RSS model is explicitly non-measured decomposition."
            ),
        },
        "claim_boundary": (
            "The file-size target arithmetic is exact for the measured Run-9 Q4_K_M file and llama-bench parameter count. "
            "The total-RSS projections are an explicitly optimistic illustrative model, not measured LARC memory."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "q4_effective_file_bpw": q4_eff_bpw,
        "targets": exact_file_targets,
        "projection_geometry": geom,
        "mixed_examples": mixed_projection_examples,
    }, indent=2))


if __name__ == "__main__":
    main()
