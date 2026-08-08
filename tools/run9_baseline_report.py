#!/usr/bin/env python3
"""Assemble Run-9 llama.cpp competitive baseline artifacts.

Input directory contains:
- metadata.json
- ppl_{F16,Q4_K_M,Q2_K}.txt
- bench_{Q4_K_M,Q2_K}_{pp,tg}.json
- rss_{Q4_K_M,Q2_K}_ctx{64,2048,8192}_{mmap,nommap}_repN.txt

The output is intentionally explicit about measured vs modeled quantities. RSS is
process MaxRSS from GNU /usr/bin/time on a GitHub-hosted Ubuntu CPU runner; it is
not consumer-device VRAM and it is not a LARC result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+\-]+)\s*\+/-\s*([0-9.eE+\-]+)")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_ppl(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    matches = PPL_RE.findall(text)
    if not matches:
        raise RuntimeError(f"No final PPL estimate in {path}")
    ppl, unc = matches[-1]
    return {"ppl": float(ppl), "uncertainty": float(unc)}


def parse_rss(path: Path) -> int:
    text = path.read_text(errors="replace")
    m = RSS_RE.search(text)
    if not m:
        raise RuntimeError(f"No GNU time MaxRSS in {path}")
    return int(m.group(1)) * 1024


def compact_bench(path: Path, kind: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise RuntimeError(f"Expected JSON list in {path}")
    rows = []
    for r in data:
        item = {
            "n_prompt": int(r.get("n_prompt", 0)),
            "n_gen": int(r.get("n_gen", 0)),
            "n_depth": int(r.get("n_depth", 0)),
            "avg_tokens_per_second": float(r["avg_ts"]),
            "stddev_tokens_per_second": float(r.get("stddev_ts", 0.0)),
            "threads": int(r.get("n_threads", 0)),
            "use_mmap": bool(r.get("use_mmap", True)),
            "backend": r.get("backends"),
            "cpu_info": r.get("cpu_info"),
            "model_size": int(r.get("model_size", 0)),
            "model_n_params": int(r.get("model_n_params", 0)),
        }
        if kind == "pp" and item["n_prompt"] <= 0:
            continue
        if kind == "tg" and item["n_gen"] <= 0:
            continue
        rows.append(item)
    return rows


def rss_summary(root: Path, quant: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ctx in (64, 2048, 8192):
        ctx_rows = {}
        for mode in ("mmap", "nommap"):
            vals = []
            for p in sorted(root.glob(f"rss_{quant}_ctx{ctx}_{mode}_rep*.txt")):
                vals.append(parse_rss(p))
            if not vals:
                raise RuntimeError(f"Missing RSS files for {quant} ctx={ctx} {mode}")
            ctx_rows[mode] = {
                "repetitions": len(vals),
                "maxrss_bytes_each": vals,
                "median_maxrss_bytes": int(statistics.median(vals)),
                "min_maxrss_bytes": min(vals),
                "max_maxrss_bytes": max(vals),
            }
        out[str(ctx)] = ctx_rows
    return out


def ratio(a: float, b: float) -> float:
    return a / b if b else math.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/run9_llamacpp_baseline.json"))
    args = ap.parse_args()

    raw = args.raw
    meta = json.loads((raw / "metadata.json").read_text())

    ppls = {q: parse_ppl(raw / f"ppl_{q}.txt") for q in ("F16", "Q4_K_M", "Q2_K")}
    results: dict[str, Any] = {}
    for q in ("Q4_K_M", "Q2_K"):
        results[q] = {
            "file": meta["models"][q],
            "perplexity": ppls[q],
            "maxrss": rss_summary(raw, q),
            "throughput": {
                "prompt_processing": compact_bench(raw / f"bench_{q}_pp.json", "pp"),
                "generation": compact_bench(raw / f"bench_{q}_tg.json", "tg"),
            },
        }

    q4_size = meta["models"]["Q4_K_M"]["bytes"]
    q2_size = meta["models"]["Q2_K"]["bytes"]
    comparisons: dict[str, Any] = {
        "q4_k_m_ppl_ratio_vs_f16": ratio(ppls["Q4_K_M"]["ppl"], ppls["F16"]["ppl"]),
        "q2_k_ppl_ratio_vs_f16": ratio(ppls["Q2_K"]["ppl"], ppls["F16"]["ppl"]),
        "q2_k_ppl_ratio_vs_q4_k_m": ratio(ppls["Q2_K"]["ppl"], ppls["Q4_K_M"]["ppl"]),
        "q4_k_m_file_reduction_vs_f16_x": ratio(meta["models"]["F16"]["bytes"], q4_size),
        "q2_k_file_reduction_vs_f16_x": ratio(meta["models"]["F16"]["bytes"], q2_size),
        "q4_k_m_to_q2_k_file_reduction_x": ratio(q4_size, q2_size),
        "rss_q4_to_q2_reduction_x": {},
    }
    for ctx in (64, 2048, 8192):
        comparisons["rss_q4_to_q2_reduction_x"][str(ctx)] = {}
        for mode in ("mmap", "nommap"):
            a = results["Q4_K_M"]["maxrss"][str(ctx)][mode]["median_maxrss_bytes"]
            b = results["Q2_K"]["maxrss"][str(ctx)][mode]["median_maxrss_bytes"]
            comparisons["rss_q4_to_q2_reduction_x"][str(ctx)][mode] = ratio(a, b)

    out = {
        "run": 9,
        "evidence_level": "measured competitive deployment baseline on GitHub-hosted CPU runner",
        "baseline": "llama.cpp Q4_K_M",
        "comparison_quant": "llama.cpp Q2_K",
        "source_model": meta["source_model"],
        "source_model_commit": meta["source_model_commit"],
        "llama_cpp": meta["llama_cpp"],
        "runner": meta["runner"],
        "wikitext2": meta["wikitext2"],
        "measurement_contract": meta["measurement_contract"],
        "f16_reference": {
            "file": meta["models"]["F16"],
            "perplexity": ppls["F16"],
        },
        "quantized": results,
        "comparisons": comparisons,
        "claim_boundary": (
            "Measured GGUF/llama.cpp baseline only. MaxRSS is process resident memory on an ephemeral "
            "GitHub-hosted CPU runner, not VRAM and not a consumer-device L4 result. No LARC quality or "
            "memory claim is established by this artifact."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "F16_PPL": ppls["F16"],
        "Q4_K_M_PPL": ppls["Q4_K_M"],
        "Q2_K_PPL": ppls["Q2_K"],
        "comparisons": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
