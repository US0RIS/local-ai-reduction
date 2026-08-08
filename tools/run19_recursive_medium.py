#!/usr/bin/env python3
"""Run 19: medium-budget global recursive recovery at the 10x capacity frontier.

Run 16/P=2/rank8 moved from an unusable initialization to 97.23x teacher PPL
after only 10,240 sampled tokens while reducing best training loss by 85.97%.
That formally failed the short-budget gate but proved global optimization changes
the behavior radically compared with local post-hoc sharing.

Run 19 holds P=2 fixed and tests the two adapter ranks that still fit the current
10x serialized description envelope when composed with Run-17 vocabulary PQ:
- rank8: Run-18 fits with PQ subdim >=12;
- rank16: exact decoder Q4 bytes fit with PQ subdim >=24.
Rank32 is intentionally excluded because it no longer fits the same conservative
10x byte envelope.

This remains a conversion/uptraining pilot, not a recursive-pretraining result.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run16_global_recursive_pilot as r16

SHORT_RUN16_P2_PPL_RATIO = 97.22893995975434
SHORT_RUN16_P2_TOKENS = 10_240


def optimizer_groups(student):
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
    return base, adapters, norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--eval-corpus", type=Path, required=True)
    ap.add_argument("--rank", type=int, choices=(8, 16), required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-sequences", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--adapter-lr", type=float, default=5e-4)
    ap.add_argument("--norm-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=32)
    ap.add_argument("--eval-predictions", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.train_corpus.resolve() == args.eval_corpus.resolve():
        raise RuntimeError("train and held-out evaluation corpora must be disjoint")

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

    train_ids = r16.token_stream(tokenizer, args.train_corpus)
    eval_ids = r16.token_stream(tokenizer, args.eval_corpus)
    student, structural = r16.build_recursive_student(teacher, physical_blocks=2, rank=args.rank)
    gc.collect()

    base, adapters, norms = optimizer_groups(student)
    optimizer = torch.optim.AdamW(
        [
            {"params": base, "lr": args.base_lr, "weight_decay": 0.0},
            {"params": adapters, "lr": args.adapter_lr, "weight_decay": 0.0},
            {"params": norms, "lr": args.norm_lr, "weight_decay": 0.0},
        ]
    )

    def lr_factor(step: int) -> float:
        # Linear warmup, then cosine decay to 10% of the initial LR.
        if step < args.warmup_steps:
            return max(1e-3, (step + 1) / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return 0.1 + 0.9 * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    t0 = time.perf_counter()
    teacher_eval = r16.evaluate_ppl(teacher, eval_ids, context=256, max_predictions=args.eval_predictions)
    initial_student = r16.evaluate_ppl(student, eval_ids, context=256, max_predictions=1024)
    teacher_initial = r16.evaluate_ppl(teacher, eval_ids, context=256, max_predictions=1024)
    initial_student["ppl_ratio_vs_teacher_same_slice"] = initial_student["ppl"] / teacher_initial["ppl"]

    max_start = int(train_ids.numel() - args.seq_len - 1)
    gen = torch.Generator().manual_seed(19000 + args.rank)
    history = []
    first_loss = None
    best_loss = math.inf
    best_step = None
    first_ce = None
    best_ce = math.inf

    student.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, max_start + 1, (args.batch_sequences,), generator=gen)
        xs = []
        ys = []
        for st in starts.tolist():
            seq = train_ids[st:st + args.seq_len + 1]
            xs.append(seq[:-1])
            ys.append(seq[1:])
        x = torch.stack(xs, dim=0)
        labels = torch.stack(ys, dim=0)

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
        scheduler.step()

        lv = float(loss.detach())
        cv = float(ce.detach())
        kv = float(kl.detach())
        if first_loss is None:
            first_loss = lv
            first_ce = cv
        if lv < best_loss:
            best_loss = lv
            best_step = step
        best_ce = min(best_ce, cv)

        if step == 1 or step % 32 == 0 or step == args.steps:
            history.append({
                "step": step,
                "loss": lv,
                "kl": kv,
                "ce": cv,
                "lr_factor": lr_factor(step),
            })

    final_student = r16.evaluate_ppl(student, eval_ids, context=256, max_predictions=args.eval_predictions)
    final_student["delta_nll_vs_teacher"] = final_student["nll"] - teacher_eval["nll"]
    final_student["ppl_ratio_vs_teacher"] = math.exp(final_student["delta_nll_vs_teacher"])

    sampled_tokens = args.steps * args.seq_len * args.batch_sequences
    best_loss_reduction = (first_loss - best_loss) / first_loss
    best_ce_reduction = (first_ce - best_ce) / first_ce
    ratio = final_student["ppl_ratio_vs_teacher"]
    reduction = structural["decoder_main_projection_parameter_reduction_x"]

    if reduction >= 8.0 and ratio <= 1.5:
        decision = "pass_medium_recursive_recovery"
    elif reduction >= 8.0 and ratio <= 10.0 and best_loss_reduction >= 0.50:
        decision = "promising_extend_long_uptraining"
    else:
        decision = "fail_medium_budget_only"

    # Description-budget compatibility is deterministic and precomputed from the
    # same Q4_GROUP64 byte formula used by Run 18.
    if args.rank == 8:
        byte_compat = {
            "minimum_run17_pq_subdim_for_10x_under_run18_allowance": 12,
            "example_subdim32_total_bytes": 8_533_440,
            "example_subdim32_reduction_vs_q4_k_m_x": 12.35770052874339,
        }
    else:
        byte_compat = {
            "minimum_run17_pq_subdim_for_10x_under_same_allowance": 24,
            "subdim24_total_bytes": 10_086_912,
            "subdim24_reduction_vs_q4_k_m_x": 10.454507385411908,
            "subdim32_total_bytes": 9_792_000,
            "subdim32_reduction_vs_q4_k_m_x": 10.769372549019607,
        }

    out = {
        "run": 19,
        "kind": "medium_budget_global_recursive_recovery",
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "structural": structural,
        "description_budget_compatibility": byte_compat,
        "training": {
            "train_corpus": str(args.train_corpus),
            "eval_corpus": str(args.eval_corpus),
            "disjoint": True,
            "steps": args.steps,
            "sequence_length": args.seq_len,
            "batch_sequences": args.batch_sequences,
            "sampled_training_tokens": sampled_tokens,
            "token_budget_multiple_vs_run16_short_p2": sampled_tokens / SHORT_RUN16_P2_TOKENS,
            "objective": "0.75 forward-KL teacher distillation at T=2 + 0.25 next-token CE",
            "optimizer": "AdamW; no weight decay",
            "schedule": "32-step linear warmup then cosine decay to 10% initial LR",
            "base_lr": args.base_lr,
            "adapter_lr": args.adapter_lr,
            "norm_lr": args.norm_lr,
            "first_loss": first_loss,
            "best_loss": best_loss,
            "best_loss_step": best_step,
            "best_loss_reduction_fraction": best_loss_reduction,
            "first_ce": first_ce,
            "best_ce": best_ce,
            "best_ce_reduction_fraction": best_ce_reduction,
            "history": history,
        },
        "quality": {
            "teacher_reference": teacher_eval,
            "initial_student_short_probe": initial_student,
            "final_student": final_student,
            "run16_short_p2_rank8_reference_ppl_ratio": SHORT_RUN16_P2_PPL_RATIO if args.rank == 8 else None,
            "improvement_vs_run16_short_p2_ppl_ratio_x": SHORT_RUN16_P2_PPL_RATIO / ratio if args.rank == 8 else None,
        },
        "precommitted_medium_gate": {
            "pass": ">=8x main-projection parameter reduction and held-out PPL ratio <=1.5x teacher",
            "promising_extend": ">=8x reduction, held-out PPL ratio <=10x teacher, and best training loss improves >=50% from first update",
            "failure_scope": "failure rejects only this 131,072-token medium conversion recipe at the tested rank; it does not falsify billion-token recursive uptraining or recursive pretraining",
        },
        "decision": decision,
        "wall_seconds": time.perf_counter() - t0,
        "claim_boundary": (
            "Real-pretrained full-model quality pilot with original tied embedding/head unchanged. Exact structural parameter counts and Run-18-compatible byte envelopes are reported separately. No packed recursive file, native runtime, RSS or VRAM claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "rank": args.rank,
        "structural_reduction_x": reduction,
        "sampled_tokens": sampled_tokens,
        "teacher_ppl": teacher_eval["ppl"],
        "initial_ratio": initial_student["ppl_ratio_vs_teacher_same_slice"],
        "final_ratio": ratio,
        "best_loss_reduction": best_loss_reduction,
        "decision": decision,
        "description_budget": byte_compat,
    }, indent=2))


if __name__ == "__main__":
    main()
