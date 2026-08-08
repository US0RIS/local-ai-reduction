#!/usr/bin/env python3
"""Run 6 partial real-model conversion experiment.

Representation-consistent test on SmolLM2-135M:
- baseline teacher: every matrix projected to canonical row-Q4; 1-D params FP16;
- student: identical baseline outside a chosen decoder-layer group;
- chosen logical layers physically alias one donor block;
- donor matrices use Q4_GROUP64 and are hard-projected after each optimizer step;
- only the donor block is trainable;
- recovery uses teacher-logit distillation on calibration windows;
- held-out NLL is evaluated against the same row-Q4 teacher baseline.

This does not yet produce a .larc file or packed full-model runtime. It tests whether
real pretrained layer sharing can be recovered locally without changing unrelated
weights or allowing the memory/quality representations to diverge.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from run6_real_model_falsification import TEXT, get_layers, nll


def row_q4_tensor(w: torch.Tensor) -> torch.Tensor:
    x = w.detach().float()
    if x.ndim != 2:
        return x.half().float()
    pos = x.amax(1).clamp_min(0) / 7.0
    neg = (-x.amin(1)).clamp_min(0) / 8.0
    sc = torch.maximum(pos, neg).clamp_min(1e-8).half().float()
    return torch.round(x / sc[:, None]).clamp(-8, 7) * sc[:, None]


def group64_q4_tensor(w: torch.Tensor, group: int = 64) -> torch.Tensor:
    x = w.detach().float()
    if x.ndim != 2:
        return x.half().float()
    out = torch.empty_like(x)
    for s in range(0, x.shape[1], group):
        a = x[:, s:s + group]
        pos = a.amax(1).clamp_min(0) / 7.0
        neg = (-a.amin(1)).clamp_min(0) / 8.0
        sc = torch.maximum(pos, neg).clamp_min(1e-8).half().float()
        out[:, s:s + group] = torch.round(a / sc[:, None]).clamp(-8, 7) * sc[:, None]
    return out


def project_model_row_q4_(m):
    with torch.no_grad():
        seen = set()
        for p in m.parameters():
            ptr = p.untyped_storage().data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            p.copy_(row_q4_tensor(p))


def project_module_group64_(m):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(group64_q4_tensor(p))


def row_q4_param_bytes(p: torch.Tensor) -> int:
    if p.ndim == 2:
        return p.shape[0] * ((p.shape[1] + 1) // 2 + 2)
    return p.numel() * 2


def group64_param_bytes(p: torch.Tensor, group: int = 64) -> int:
    if p.ndim == 2:
        groups = math.ceil(p.shape[1] / group)
        return p.shape[0] * ((p.shape[1] + 1) // 2 + groups * 2)
    return p.numel() * 2


def unique_model_bytes(m, grouped_module=None) -> int:
    grouped_ids = {id(p) for p in grouped_module.parameters()} if grouped_module else set()
    seen = set(); total = 0
    for p in m.parameters():
        ptr = p.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += group64_param_bytes(p) if id(p) in grouped_ids else row_q4_param_bytes(p)
    return total


def parse_group(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--group', default='14,15,16,17')
    ap.add_argument('--donor', type=int, default=15)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--train-windows', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=Path, default=Path('benchmarks/run6_partial_real_conversion.json'))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(6); torch.manual_seed(6)
    tok = AutoTokenizer.from_pretrained(args.model)
    base_fp = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval().cpu()
    teacher = copy.deepcopy(base_fp).eval()
    project_model_row_q4_(teacher)

    student = copy.deepcopy(teacher).eval()
    layers = get_layers(student)
    group = parse_group(args.group)
    if args.donor not in group or max(group) >= len(layers):
        raise ValueError('invalid group/donor')

    donor = layers[args.donor]
    originals = {i: layers[i] for i in group}
    for i in group:
        layers[i] = donor
    project_module_group64_(donor)

    for p in student.parameters(): p.requires_grad_(False)
    for p in donor.parameters(): p.requires_grad_(True)

    ids = tok(TEXT, return_tensors='pt', add_special_tokens=False).input_ids
    # Disjoint calibration/evaluation halves.
    split = ids.shape[1] // 2
    train_ids = ids[:, :split]
    eval_ids = ids[:, split:]
    if eval_ids.shape[1] > 257: eval_ids = eval_ids[:, :257]

    baseline_nll = nll(teacher, eval_ids)
    fp32_nll = nll(base_fp, eval_ids)
    pre_nll = nll(student, eval_ids)

    # Fixed calibration windows and teacher logits. This avoids a teacher forward
    # on every recovery step and keeps the run deterministic.
    starts = torch.linspace(0, max(0, train_ids.shape[1] - args.window - 1), args.train_windows).long().tolist()
    examples = []
    with torch.inference_mode():
        for st in starts:
            x = train_ids[:, st:st + args.window]
            tlog = teacher(input_ids=x, use_cache=False).logits.detach().float()
            examples.append((x, tlog))

    opt = torch.optim.AdamW(donor.parameters(), lr=args.lr)
    curve = []
    student.train()
    for step in range(args.steps):
        x, tlog = examples[step % len(examples)]
        slog = student(input_ids=x, use_cache=False).logits.float()
        # Forward KL teacher -> student, temperature 1.0.
        teacher_p = torch.softmax(tlog, -1)
        loss = F.kl_div(torch.log_softmax(slog, -1), teacher_p, reduction='batchmean') / x.shape[1]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(donor.parameters(), 1.0)
        opt.step(); project_module_group64_(donor)
        if step == 0 or (step + 1) % 4 == 0:
            student.eval(); cur = nll(student, eval_ids); student.train()
            curve.append({'step': step + 1, 'distill_loss': float(loss.detach()), 'heldout_nll': cur})
    student.eval()
    post_nll = nll(student, eval_ids)

    teacher_bytes = unique_model_bytes(teacher)
    student_bytes = unique_model_bytes(student, grouped_module=donor)
    original_group_bytes = sum(row_q4_param_bytes(p) for i in group for p in originals[i].parameters())
    shared_group_bytes = sum(group64_param_bytes(p) for p in donor.parameters())

    out = {
        'run': 6,
        'evidence_level': 'L3-precheck partial real pretrained conversion',
        'model': args.model,
        'model_commit': getattr(base_fp.config, '_commit_hash', None),
        'group': group,
        'donor': args.donor,
        'logical_layers_in_group': len(group),
        'physical_blocks_for_group': 1,
        'protocol': {
            'baseline': 'all matrices canonical row-Q4; 1-D FP16',
            'student_outside_group': 'same row-Q4 baseline',
            'shared_block': 'Q4_GROUP64 hard-projected after every update',
            'trainable_parameters': 'shared donor block only',
            'recovery': 'teacher-logit KL distillation',
            'steps': args.steps,
            'lr': args.lr,
            'window': args.window,
            'train_windows': args.train_windows,
            'calibration_and_eval_disjoint': True,
        },
        'quality': {
            'fp32_teacher_nll': fp32_nll,
            'row_q4_teacher_nll': baseline_nll,
            'shared_pre_recovery_nll': pre_nll,
            'shared_post_recovery_nll': post_nll,
            'post_delta_nats_per_token_vs_row_q4': post_nll - baseline_nll,
            'post_perplexity_ratio_vs_row_q4': math.exp(post_nll - baseline_nll),
            'post_perplexity_ratio_vs_fp32': math.exp(post_nll - fp32_nll),
            'curve': curve,
        },
        'weight_accounting': {
            'row_q4_teacher_modeled_weight_bytes': teacher_bytes,
            'partial_shared_student_modeled_weight_bytes': student_bytes,
            'whole_model_weight_reduction_x': teacher_bytes / student_bytes,
            'original_group_row_q4_bytes': original_group_bytes,
            'shared_group_group64_q4_bytes': shared_group_bytes,
            'group_weight_reduction_x': original_group_bytes / shared_group_bytes,
        },
        'claim_boundary': 'Partial real-model sharing/recovery experiment only; no complete LARC model, KV compression, measured RSS, or packed full-model runtime.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
