#!/usr/bin/env python3
"""Run 6: real-pretrained-model falsification harness.

Default target: HuggingFaceTB/SmolLM2-135M.

Measures on a real pretrained decoder-only Transformer:
  1. held-out activation-subspace energy and *operator-output* NMSE for
     q/k/v/o/gate/up/down projection inputs at several ranks;
  2. raw functional interchangeability of real decoder layers by physically
     aliasing one layer module into another logical depth and measuring NLL;
  3. small contiguous exact-sharing groups, before any recovery training.

The script intentionally does not claim a LARC conversion. It is a falsification
instrument: if real activations are not compressible at useful ranks, or raw
cross-depth function sharing is catastrophically bad, later conversion work must
change direction before spending more effort on packing/runtime optimization.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

SITE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_LAYERS = (0, 5, 10, 15, 20, 25, 29)
DEFAULT_RANKS = (8, 16, 32, 64, 128)

# Original calibration prose; no benchmark corpus dependency is required.
TEXT = """
A city changes gradually when transportation, housing, and employment evolve at
unequal rates. A new rail line can shorten a commute while increasing the value
of land near a station. The effect is rarely uniform: some neighborhoods gain
new businesses, while others face congestion or higher rents. Good policy must
therefore distinguish average outcomes from the distribution of outcomes.

Scientific models are useful because they compress observations into rules that
can be tested. A model is not valuable merely because it reproduces its training
examples. It should make accurate predictions on observations that were not used
to choose its parameters. When two explanations fit the same evidence, a new
experiment should be designed to make their predictions diverge.

Computer systems have similar tradeoffs. Caches reduce repeated work but consume
memory; compression reduces memory but may add decoding cost; parallelism raises
throughput but can increase synchronization overhead. An engineering design is
credible only when all of these costs are counted under the same workload.

History is full of technologies whose early advantages appeared in narrow tests
but disappeared at larger scale. Other technologies showed the opposite pattern:
a simple prototype exposed a mechanism that became more useful as implementation
improved. Distinguishing these cases requires measurements that attack the core
assumption rather than polishing the prototype indefinitely.

In language, meaning depends on relationships across many scales. A pronoun may
refer to a noun several sentences earlier; a technical term may depend on a
definition introduced at the beginning of a document. Local regularities still
matter, but a capable model must combine them with longer-range structure.

Suppose a laboratory measures a signal repeatedly. Random error can be reduced by
additional samples, but systematic error survives averaging. If a measuring
instrument is biased, collecting ten times more data can make the wrong answer
look more precise. The correct response is to test the instrument against an
independent reference and redesign the measurement process if necessary.

A small business deciding whether to expand has to reason about uncertain demand,
fixed costs, staffing, inventory, and financing. The most optimistic scenario is
not a forecast. A useful plan includes ranges, identifies assumptions that drive
the result, and states which observations would cause the decision to change.

Software interfaces also encode assumptions. When a program treats logical
objects and physical storage as identical, it may duplicate data unnecessarily.
Separating the logical graph from the physical representation can create large
savings, but only if the shared representation preserves the behavior required by
the logical objects that reference it.
""" * 5


def parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def get_layers(model):
    # SmolLM2 / LlamaForCausalLM path, with small compatibility fallbacks.
    for path in (("model", "layers"), ("model", "model", "layers"), ("transformer", "h")):
        obj = model
        try:
            for name in path:
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            pass
    raise RuntimeError("Could not locate decoder layer ModuleList")


def find_projection(layer, name: str):
    for prefix in ("self_attn", "mlp"):
        parent = getattr(layer, prefix, None)
        if parent is not None and hasattr(parent, name):
            return getattr(parent, name)
    raise AttributeError(name)


def nll(model, ids: torch.Tensor) -> float:
    with torch.inference_mode():
        out = model(input_ids=ids[:, :-1], use_cache=False)
        logits = out.logits.float()
        target = ids[:, 1:]
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))


def collect_inputs(model, ids: torch.Tensor, selected_layers: tuple[int, ...]):
    layers = get_layers(model)
    buckets: dict[str, list[torch.Tensor]] = {}
    handles = []
    for li in selected_layers:
        if li >= len(layers):
            continue
        layer = layers[li]
        for site in SITE_NAMES:
            try:
                mod = find_projection(layer, site)
            except AttributeError:
                continue
            key = f"layer.{li}.{site}"
            buckets[key] = []
            def hook(_mod, inp, _out, key=key):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
                buckets[key].append(x)
            handles.append(mod.register_forward_hook(hook))
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0) for k, v in buckets.items() if v}


def projection_metrics(module, x_cal: torch.Tensor, x_eval: torch.Tensor, ranks: tuple[int, ...]):
    # SVD of calibration activations. Because rows=tokens and columns=features,
    # right singular vectors are the candidate activation input basis.
    _, s, vh = torch.linalg.svd(x_cal, full_matrices=False)
    energy_total = float((s * s).sum().clamp_min(1e-30))
    w = module.weight.detach().float().cpu()
    y = x_eval @ w.t()
    y_power = float((y * y).sum().clamp_min(1e-30))
    rows = []
    max_rank = vh.shape[0]
    for requested in ranks:
        r = min(requested, max_rank)
        b = vh[:r]
        x_proj = (x_eval @ b.t()) @ b
        yp = x_proj @ w.t()
        err = float(((yp - y) ** 2).sum() / y_power)
        e = float((s[:r] * s[:r]).sum() / energy_total)
        rows.append({
            "requested_rank": requested,
            "effective_rank": r,
            "input_dim": x_cal.shape[1],
            "rank_fraction": r / x_cal.shape[1],
            "calibration_energy_fraction": e,
            "heldout_operator_output_nmse": err,
        })
    return rows


def evaluate_alias(model, ids, targets: list[int], donor: int) -> float:
    layers = get_layers(model)
    originals = {i: layers[i] for i in targets}
    donor_module = layers[donor]
    try:
        for i in targets:
            layers[i] = donor_module
        return nll(model, ids)
    finally:
        for i, mod in originals.items():
            layers[i] = mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/run6_real_model_falsification.json"))
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    ap.add_argument("--cal-tokens", type=int, default=256)
    ap.add_argument("--eval-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval().cpu()
    layers = get_layers(model)

    selected = tuple(i for i in parse_csv_ints(args.layers) if i < len(layers))
    ranks = parse_csv_ints(args.ranks)
    tok = tokenizer(TEXT, return_tensors="pt", add_special_tokens=False).input_ids
    needed = args.cal_tokens + args.eval_tokens + 1
    if tok.shape[1] < needed:
        raise RuntimeError(f"Calibration text yielded {tok.shape[1]} tokens; need {needed}")
    cal = tok[:, :args.cal_tokens]
    ev = tok[:, args.cal_tokens:needed]

    cal_inputs = collect_inputs(model, cal, selected)
    eval_inputs = collect_inputs(model, ev[:, :-1], selected)

    spectra = {}
    for key, xc in cal_inputs.items():
        xe = eval_inputs[key]
        _, li, site = key.split(".")
        mod = find_projection(layers[int(li)], site)
        spectra[key] = projection_metrics(mod, xc, xe, ranks)

    base_nll = nll(model, ev)

    # Neighbor substitutions at representative depths: a direct functional test
    # of whether two real pretrained blocks can be physically aliased *without*
    # recovery. This is deliberately harsh and therefore falsification-oriented.
    alias_single = []
    for target in selected[1:-1]:
        donor = target - 1
        aliased = evaluate_alias(model, ev, [target], donor)
        alias_single.append({
            "target_layer": target,
            "donor_layer": donor,
            "nll": aliased,
            "delta_nats_per_token": aliased - base_nll,
            "perplexity_ratio": math.exp(aliased - base_nll),
        })

    alias_groups = []
    for group in ([8, 9], [14, 15, 16, 17], [22, 23, 24, 25]):
        if max(group) >= len(layers):
            continue
        donor = group[len(group)//2]
        targets = [i for i in group if i != donor]
        aliased = evaluate_alias(model, ev, targets, donor)
        alias_groups.append({
            "logical_layers": group,
            "physical_donor_layer": donor,
            "physical_blocks_for_group": 1,
            "nll": aliased,
            "delta_nats_per_token": aliased - base_nll,
            "perplexity_ratio": math.exp(aliased - base_nll),
        })

    # Compact decision summaries at rank 32 and 64.
    summaries = {}
    for rank in (32, 64):
        vals = []
        for rows in spectra.values():
            match = next((r for r in rows if r["requested_rank"] == rank), None)
            if match:
                vals.append(match["heldout_operator_output_nmse"])
        if vals:
            summaries[str(rank)] = {
                "sites": len(vals),
                "median_output_nmse": statistics.median(vals),
                "mean_output_nmse": statistics.mean(vals),
                "fraction_below_0.01": sum(v < .01 for v in vals) / len(vals),
                "fraction_below_0.03": sum(v < .03 for v in vals) / len(vals),
                "fraction_below_0.05": sum(v < .05 for v in vals) / len(vals),
            }

    out = {
        "run": 6,
        "evidence_level": "L3-precheck real pretrained model activation/function falsification",
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "architecture": {
            "layers": len(layers),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "intermediate_size": getattr(model.config, "intermediate_size", None),
            "attention_heads": getattr(model.config, "num_attention_heads", None),
            "kv_heads": getattr(model.config, "num_key_value_heads", None),
            "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        },
        "protocol": {
            "selected_layers": selected,
            "ranks": ranks,
            "calibration_tokens": int(cal.shape[1]),
            "evaluation_tokens": int(ev.shape[1] - 1),
            "activation_basis_fit": "SVD right singular vectors on calibration projection inputs",
            "projection_quality": "held-out exact linear-operator output NMSE after input projection",
            "sharing_quality": "held-out autoregressive NLL after exact physical decoder-layer aliasing; no recovery",
        },
        "baseline_nll": base_nll,
        "projection_summary": summaries,
        "projection_sites": spectra,
        "single_layer_alias": alias_single,
        "contiguous_group_alias": alias_groups,
        "claim_boundary": "Real pretrained-model diagnostic only. No LARC recovery, packed conversion, memory reduction, or hardware claim is established by this artifact.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "model": out["model"],
        "baseline_nll": base_nll,
        "projection_summary": summaries,
        "single_layer_alias": alias_single,
        "contiguous_group_alias": alias_groups,
    }, indent=2))


if __name__ == "__main__":
    main()
