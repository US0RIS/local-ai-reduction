#!/usr/bin/env python3
"""Run 18: exact tensor-description budget envelope for candidate components.

This is arithmetic, not a quality result. It combines:
- measured Run-9A Q4_K_M file size and exact parameter count;
- Run-16 recursive projection geometry (P=2/3, rank8 depth LoRA);
- Run-17 vocabulary-PQ byte contracts;
- all remaining unique SmolLM2 parameters stored as FP16;
- a conservative non-tensor allowance equal to the entire measured F16 GGUF
  surplus above raw FP16 unique-parameter bytes.

The purpose is to answer a narrow question before Run 16/17 quality results:
if those component representations actually preserve intelligence, are their
serialized-byte economics sufficient for the 10x Q4_K_M target?
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PARAMS = 134_515_008
Q4_K_M_FILE = 105_453_696
F16_FILE = 270_885_504
VOCAB = 49_152
HIDDEN = 576
KV = 192
FF = 1_536
LAYERS = 30
RANK = 8
GROUP = 64

SITES = (
    (HIDDEN, HIDDEN), # q
    (KV, HIDDEN),     # k
    (KV, HIDDEN),     # v
    (HIDDEN, HIDDEN), # o
    (FF, HIDDEN),     # gate
    (FF, HIDDEN),     # up
    (HIDDEN, FF),     # down
)


def q4_group64_bytes(rows: int, cols: int) -> int:
    total = 0
    for s in range(0, cols, GROUP):
        width = min(GROUP, cols - s)
        total += rows * (math.ceil(width / 2) + 2) # nibbles + FP16 scale
    return total


def recursive_projection_bytes(physical_phases: int, rank: int = RANK) -> dict[str, int]:
    bases = physical_phases * sum(q4_group64_bytes(o, i) for o, i in SITES)
    adapters = LAYERS * sum(
        q4_group64_bytes(rank, i) + q4_group64_bytes(o, rank)
        for o, i in SITES
    )
    return {"shared_base_q4_bytes": bases, "depth_lora_q4_bytes": adapters, "total": bases + adapters}


def vocab_pq_bytes(subdim: int) -> dict[str, int]:
    if HIDDEN % subdim:
        raise ValueError(subdim)
    m = HIDDEN // subdim
    codebooks = m * 256 * subdim * 2
    codes = VOCAB * m
    norms = VOCAB * 2
    return {"subspaces": m, "codebooks_fp16": codebooks, "codes_uint8": codes, "norms_fp16": norms, "total": codebooks + codes + norms}


def main() -> None:
    embedding_params = VOCAB * HIDDEN
    main_projection_params = LAYERS * sum(o * i for o, i in SITES)
    remaining_params = PARAMS - embedding_params - main_projection_params
    if remaining_params != 35_136:
        raise RuntimeError(f"unexpected remaining parameter count: {remaining_params}")

    remaining_fp16 = remaining_params * 2
    raw_fp16_unique_tensor_bytes = PARAMS * 2
    conservative_non_tensor_allowance = F16_FILE - raw_fp16_unique_tensor_bytes
    if conservative_non_tensor_allowance < 0:
        raise RuntimeError("F16 GGUF smaller than raw unique FP16 tensors")

    target10 = Q4_K_M_FILE / 10
    independent_projection_q4 = LAYERS * sum(q4_group64_bytes(o, i) for o, i in SITES)
    tied_vocab_q4 = q4_group64_bytes(VOCAB, HIDDEN)

    configs = []
    for p in (2, 3):
        rec = recursive_projection_bytes(p)
        for subdim in (8, 12, 16, 24, 32):
            vocab = vocab_pq_bytes(subdim)
            tensor_payload = rec["total"] + vocab["total"] + remaining_fp16
            conservative_total = tensor_payload + conservative_non_tensor_allowance
            configs.append({
                "physical_projection_phases": p,
                "depth_lora_rank": RANK,
                "vocab_pq_subdim": subdim,
                "recursive_projection": rec,
                "vocabulary": vocab,
                "remaining_unique_params_fp16_bytes": remaining_fp16,
                "tensor_payload_bytes": tensor_payload,
                "conservative_non_tensor_allowance_bytes": conservative_non_tensor_allowance,
                "conservative_total_description_bytes": conservative_total,
                "headroom_to_10x_target_bytes": target10 - conservative_total,
                "modeled_file_reduction_vs_measured_q4_k_m_x": Q4_K_M_FILE / conservative_total,
                "fits_10x_serialized_file_budget": conservative_total <= target10,
            })

    out = {
        "run": 18,
        "kind": "description_budget_envelope_arithmetic",
        "measured_external_inputs": {
            "q4_k_m_file_bytes": Q4_K_M_FILE,
            "f16_gguf_file_bytes": F16_FILE,
            "unique_parameter_count": PARAMS,
            "ten_x_total_file_budget_bytes": target10,
        },
        "model_geometry": {
            "tied_embedding_parameters": embedding_params,
            "main_decoder_projection_parameters": main_projection_params,
            "remaining_unique_parameters": remaining_params,
            "independent_main_projection_q4_group64_bytes": independent_projection_q4,
            "tied_vocab_q4_group64_bytes": tied_vocab_q4,
        },
        "conservative_allowance": {
            "raw_fp16_unique_tensor_bytes": raw_fp16_unique_tensor_bytes,
            "f16_gguf_surplus_bytes": conservative_non_tensor_allowance,
            "interpretation": "Treat the entire measured F16 GGUF surplus over raw unique FP16 tensor bytes as a conservative allowance for tokenizer/container/tensor-metadata/alignment. This is not a claim that LARC will use GGUF metadata or exactly this overhead."
        },
        "configs": configs,
        "ten_x_feasible_configurations": [
            {"physical_projection_phases": c["physical_projection_phases"], "vocab_pq_subdim": c["vocab_pq_subdim"], "conservative_total_description_bytes": c["conservative_total_description_bytes"], "modeled_reduction_x": c["modeled_file_reduction_vs_measured_q4_k_m_x"], "headroom_bytes": c["headroom_to_10x_target_bytes"]}
            for c in configs if c["fits_10x_serialized_file_budget"]
        ],
        "decision": {
            "serialized_byte_economics_can_reach_10x_if_components_pass_quality": any(c["fits_10x_serialized_file_budget"] for c in configs),
            "minimum_p2_vocab_setting_that_fits": 12,
            "minimum_p3_vocab_setting_that_fits": 32,
            "interpretation": "The current structural byte contracts are sufficient in principle for 10x serialized size without requiring sub-1-bit physical decoder weights. Quality, native runtime, and measured resident memory remain completely open."
        },
        "claim_boundary": "Deterministic byte arithmetic only. Run 16 and Run 17 are pending quality experiments, Q4_GROUP64 is not Q4_K_M, no LARC file exists, and no RSS/VRAM/speed reduction is measured."
    }
    Path("benchmarks/RUN18_DESCRIPTION_BUDGET.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["ten_x_feasible_configurations"], indent=2))


if __name__ == "__main__":
    main()
