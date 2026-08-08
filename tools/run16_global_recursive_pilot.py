#!/usr/bin/env python3
"""Run 16: globally distill a highly recursive SmolLM2 decoder pilot.

This is the first structural experiment that trains the *whole language model*
after imposing aggressive parameter sharing. It is intentionally a short-budget
pilot, not a final falsification of recursive architectures.

Representation
--------------
For each of the seven large projection families (q/k/v/o/gate/up/down), only P
physical full-rank base matrices exist, where P is 2 or 3. Logical layer l uses
base p = l mod P plus a unique rank-r LoRA delta:

    W_l(x) = B_{p(l)} x + U_l V_l x

The physical bases are initialized from the arithmetic mean of their assigned
teacher matrices. Layer-specific LoRA factors are initialized by randomized SVD
of each teacher residual W_l - B_p, which is substantially closer to the
pretrained teacher than zero-LoRA initialization.

Original tied embeddings/head remain unchanged and frozen to isolate decoder
sharing viability. Original per-layer RMSNorm weights remain layer-specific and
are trainable because their storage cost is negligible.

The student is then trained end-to-end against real SmolLM2 teacher logits using
forward KL distillation plus next-token CE on WikiText-2 train tokens. Quality is
measured as full-model held-out next-token PPL on a fixed WikiText-2 test slice.

No low-bit packing is applied in Run 16. If the FP32 recursive student cannot
recover under the short budget, quantization cannot rescue it. If it does recover,
a later run can quantize the physical bases/adapters and combine the result with
the independent embedding/head work.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 30


class SharedLoRALinear(torch.nn.Module):
    def __init__(self, shared_weight: torch.nn.Parameter, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.weight = shared_weight
        self.lora_A = torch.nn.Parameter(a.contiguous().float())  # [r, in]
        self.lora_B = torch.nn.Parameter(b.contiguous().float())  # [out, r]
        self.in_features = shared_weight.shape[1]
        self.out_features = shared_weight.shape[0]
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + delta


def module_for(layer, site: str):
    return layer.self_attn if site in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp


def lowrank_residual(delta: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized SVD factors B@A ~= delta with B[out,r], A[r,in]."""
    out_dim, in_dim = delta.shape
    r = min(rank, out_dim, in_dim)
    if r <= 0:
        return torch.zeros((0, in_dim)), torch.zeros((out_dim, 0))
    # torch.svd_lowrank uses the global RNG. Preserve deterministic initialization
    # per logical matrix without making the training sampler depend on it.
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        q = min(min(out_dim, in_dim), r + 4)
        u, s, v = torch.svd_lowrank(delta.float(), q=q, niter=2)
    finally:
        torch.random.set_rng_state(state)
    u = u[:, :r]
    s = s[:r].clamp_min(0)
    v = v[:, :r]
    root = s.sqrt()
    b = u * root[None, :]
    a = root[:, None] * v.T
    return a.contiguous(), b.contiguous()


def build_recursive_student(teacher, physical_blocks: int, rank: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    if N_LAYERS % physical_blocks != 0:
        raise ValueError("physical-block count must divide 30")

    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(False)

    # Average one full-rank base per projection family and recurrence phase.
    shared: dict[tuple[str, int], torch.nn.Parameter] = {}
    base_param_count = 0
    for site in SITES:
        for phase in range(physical_blocks):
            assigned = list(range(phase, N_LAYERS, physical_blocks))
            ws = [getattr(module_for(teacher.model.layers[li], site), site).weight.detach().float() for li in assigned]
            mean = torch.stack(ws, dim=0).mean(0)
            p = torch.nn.Parameter(mean.contiguous())
            shared[(site, phase)] = p
            base_param_count += p.numel()

    adapter_param_count = 0
    residual_init_nmse: list[float] = []
    for li in range(N_LAYERS):
        phase = li % physical_blocks
        t_layer = teacher.model.layers[li]
        s_layer = student.model.layers[li]
        for si, site in enumerate(SITES):
            t_mod = getattr(module_for(t_layer, site), site)
            base = shared[(site, phase)]
            delta = t_mod.weight.detach().float() - base.detach()
            a, b = lowrank_residual(delta, rank, seed=100000 + li * 101 + si)
            approx = b @ a
            den = float(delta.square().sum())
            residual_init_nmse.append(float((approx - delta).square().sum()) / max(den, 1e-30))
            wrapper = SharedLoRALinear(base, a, b)
            setattr(module_for(s_layer, site), site, wrapper)
            adapter_param_count += a.numel() + b.numel()

        # Keep depth-specific norms; their parameter cost is tiny and they provide
        # a cheap depth-conditioning channel during global uptraining.
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(s_layer, norm_name)
            for p in norm.parameters():
                p.requires_grad_(True)

    # Ensure shared bases and all adapters are trainable.
    for li in range(N_LAYERS):
        s_layer = student.model.layers[li]
        for site in SITES:
            w = getattr(module_for(s_layer, site), site)
            w.weight.requires_grad_(True)
            w.lora_A.requires_grad_(True)
            w.lora_B.requires_grad_(True)

    # Isolate decoder sharing: tied vocabulary matrix stays original and frozen.
    student.model.embed_tokens.weight.requires_grad_(False)
    student.lm_head.weight.requires_grad_(False)

    independent_projection_params = 0
    for li in range(N_LAYERS):
        for site in SITES:
            independent_projection_params += getattr(module_for(teacher.model.layers[li], site), site).weight.numel()

    # named_parameters removes shared-Parameter duplicates by default.
    student_unique = sum(p.numel() for p in student.parameters())
    teacher_unique = sum(p.numel() for p in teacher.parameters())
    trainable_unique = sum(p.numel() for p in student.parameters() if p.requires_grad)

    meta = {
        "physical_projection_base_count_per_operator": physical_blocks,
        "logical_layers_per_recurrence_phase": N_LAYERS // physical_blocks,
        "adapter_rank": rank,
        "independent_teacher_main_projection_parameters": independent_projection_params,
        "physical_shared_base_parameters": base_param_count,
        "depth_lora_parameters": adapter_param_count,
        "decoder_main_projection_parameter_reduction_x": independent_projection_params / (base_param_count + adapter_param_count),
        "teacher_unique_parameters": teacher_unique,
        "student_unique_parameters_with_original_tied_embedding": student_unique,
        "whole_unique_parameter_reduction_x_with_original_embedding": teacher_unique / student_unique,
        "trainable_unique_parameters": trainable_unique,
        "residual_svd_init_nmse": {
            "median": statistics.median(residual_init_nmse),
            "mean": statistics.fmean(residual_init_nmse),
            "p90": float(torch.quantile(torch.tensor(residual_init_nmse), 0.9)),
        },
    }
    return student, meta


def token_stream(tokenizer, path: Path) -> torch.Tensor:
    text = path.read_text(errors="replace")
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].contiguous()


def evaluate_ppl(model, ids: torch.Tensor, context: int, max_predictions: int) -> dict[str, float]:
    model.eval()
    nll_sum = 0.0
    total = 0
    with torch.inference_mode():
        for start in range(0, min(ids.numel() - 1, max_predictions), context):
            remain = min(context, max_predictions - total)
            seq = ids[start:start + remain + 1]
            if seq.numel() < 2:
                break
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            logits = model(input_ids=x, use_cache=False).logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            nll_sum += float(loss)
            total += int(y.numel())
            if total >= max_predictions:
                break
    nll = nll_sum / total
    return {"predicted_tokens": total, "nll": nll, "ppl": math.exp(nll)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--eval-corpus", type=Path, required=True)
    ap.add_argument("--physical-blocks", type=int, choices=(2, 3), required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--adapter-lr", type=float, default=5e-4)
    ap.add_argument("--norm-lr", type=float, default=1e-4)
    ap.add_argument("--eval-predictions", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.train_corpus.resolve() == args.eval_corpus.resolve():
        raise RuntimeError("Training/distillation and held-out evaluation files must be disjoint")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    ).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_ids = token_stream(tokenizer, args.train_corpus)
    eval_ids = token_stream(tokenizer, args.eval_corpus)
    if train_ids.numel() < args.seq_len + 2:
        raise RuntimeError("Training corpus too short")

    t0 = time.perf_counter()
    student, structural = build_recursive_student(teacher, args.physical_blocks, args.rank)
    gc.collect()

    # Separate optimizer groups by role while de-duplicating shared parameters.
    base, adapters, norms = [], [], []
    seen = set()
    for name, p in student.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if "lora_A" in name or "lora_B" in name:
            adapters.append(p)
        elif "layernorm" in name:
            norms.append(p)
        else:
            base.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": base, "lr": args.base_lr, "weight_decay": 0.0},
            {"params": adapters, "lr": args.adapter_lr, "weight_decay": 0.0},
            {"params": norms, "lr": args.norm_lr, "weight_decay": 0.0},
        ]
    )

    # A small initial held-out probe shows how much the structural initialization
    # alone preserves before any global distillation.
    teacher_eval = evaluate_ppl(teacher, eval_ids, context=256, max_predictions=args.eval_predictions)
    initial_student = evaluate_ppl(student, eval_ids, context=256, max_predictions=min(1024, args.eval_predictions))
    teacher_initial_ref = evaluate_ppl(teacher, eval_ids, context=256, max_predictions=min(1024, args.eval_predictions))
    initial_student["ppl_ratio_vs_teacher_same_slice"] = initial_student["ppl"] / teacher_initial_ref["ppl"]

    student.train()
    gen = torch.Generator().manual_seed(12345 + args.physical_blocks)
    history: list[dict[str, float]] = []
    first_loss = None
    best_loss = math.inf
    max_start = int(train_ids.numel() - args.seq_len - 1)
    for step in range(1, args.steps + 1):
        start = int(torch.randint(0, max_start + 1, (1,), generator=gen))
        seq = train_ids[start:start + args.seq_len + 1]
        x = seq[:-1].unsqueeze(0)
        labels = seq[1:].unsqueeze(0)

        with torch.inference_mode():
            teacher_logits = teacher(input_ids=x, use_cache=False).logits.float()
        student_logits = student(input_ids=x, use_cache=False).logits.float()

        t = args.temperature
        kl = F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="none",
        ).sum(dim=-1).mean() * (t * t)
        ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), labels.reshape(-1))
        loss = 0.75 * kl + 0.25 * ce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()

        lv = float(loss.detach())
        kv = float(kl.detach())
        cv = float(ce.detach())
        if first_loss is None:
            first_loss = lv
        best_loss = min(best_loss, lv)
        if step == 1 or step % 10 == 0 or step == args.steps:
            history.append({"step": step, "loss": lv, "kl": kv, "ce": cv})

    final_student = evaluate_ppl(student, eval_ids, context=256, max_predictions=args.eval_predictions)
    final_student["delta_nll_vs_teacher"] = final_student["nll"] - teacher_eval["nll"]
    final_student["ppl_ratio_vs_teacher"] = math.exp(final_student["delta_nll_vs_teacher"])
    loss_reduction = (first_loss - history[-1]["loss"]) / first_loss if first_loss else 0.0
    best_loss_reduction = (first_loss - best_loss) / first_loss if first_loss else 0.0

    red = structural["decoder_main_projection_parameter_reduction_x"]
    ratio = final_student["ppl_ratio_vs_teacher"]
    if red >= 8.0 and ratio <= 1.5:
        decision = "pass_short_budget_recursive_pilot"
    elif red >= 8.0 and ratio <= 3.0 and best_loss_reduction >= 0.40:
        decision = "promising_extend_uptraining"
    else:
        decision = "fail_short_budget_pilot_only"

    out = {
        "run": 16,
        "kind": "global_recursive_depth_lora_distillation_pilot",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "structural": structural,
        "training": {
            "train_corpus": str(args.train_corpus),
            "eval_corpus": str(args.eval_corpus),
            "disjoint": True,
            "steps": args.steps,
            "tokens_per_step": args.seq_len,
            "maximum_sampled_training_tokens": args.steps * args.seq_len,
            "objective": "0.75 forward-KL teacher distillation + 0.25 next-token CE",
            "temperature": args.temperature,
            "base_lr": args.base_lr,
            "adapter_lr": args.adapter_lr,
            "norm_lr": args.norm_lr,
            "first_loss": first_loss,
            "final_logged_loss": history[-1]["loss"],
            "best_loss": best_loss,
            "final_vs_first_loss_reduction_fraction": loss_reduction,
            "best_vs_first_loss_reduction_fraction": best_loss_reduction,
            "history": history,
        },
        "quality": {
            "teacher_reference": teacher_eval,
            "initial_student_short_probe": initial_student,
            "final_student": final_student,
        },
        "precommitted_pilot_gate": {
            "pass": "decoder main-projection parameter reduction >=8x AND held-out PPL ratio <=1.5x teacher",
            "promising_extend": ">=8x reduction, PPL ratio <=3x, and best training loss improves >=40% from first step",
            "failure_scope": "A failure rejects only this <=10,240-token short-budget conversion recipe; it does not falsify long uptraining or recursively pretrained architectures.",
        },
        "decision": decision,
        "wall_seconds": time.perf_counter() - t0,
        "research_context": {
            "relation_to_prior_runs": "Unlike Runs 6/7/13/14, Run 16 globally trains the full shared decoder under language-model distillation after imposing recurrence.",
            "compression_relevance": "rank8 is intentionally selected because rank64-512 depth adapters would surrender most of the 10x-class structural gain on this 135M model.",
        },
        "claim_boundary": (
            "Real-pretrained full-model quality pilot with original tied embedding/head left unchanged. Structural parameter counts are exact for the in-memory shared Parameter graph, but no packed file, native recursive runtime, RSS or VRAM is measured. "
            "A short-budget failure must not be generalized to the recursive-architecture class; literature-scale conversions use orders of magnitude more uptraining tokens."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "physical_blocks": args.physical_blocks,
        "structural": structural,
        "teacher": teacher_eval,
        "initial_student": initial_student,
        "final_student": final_student,
        "decision": decision,
        "loss_reduction": loss_reduction,
        "best_loss_reduction": best_loss_reduction,
    }, indent=2))


if __name__ == "__main__":
    main()
