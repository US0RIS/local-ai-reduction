#!/usr/bin/env python3
"""Collect raw llama.cpp baseline outputs into one machine-readable artifact."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

PPL_RE=re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)(?:\s*\+/-\s*([0-9.eE+-]+))?")
RSS_RE=re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
ELAPSED_RE=re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)")

def parse_time(p:Path):
    t=p.read_text(errors='replace');m=RSS_RE.search(t);e=ELAPSED_RE.search(t)
    if not m: raise RuntimeError(f'no RSS in {p}')
    return {'max_rss_kib':int(m.group(1)),'max_rss_bytes':int(m.group(1))*1024,'elapsed':e.group(1).strip() if e else None}

def parse_ppl(p:Path):
    t=p.read_text(errors='replace');ms=list(PPL_RE.finditer(t))
    if not ms: raise RuntimeError(f'no final PPL in {p}')
    m=ms[-1];return {'ppl':float(m.group(1)),'stderr':float(m.group(2)) if m.group(2) else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    meta=json.load(open(a.raw/'meta.json'))
    out={'run':9,'evidence_level':'L4-ish hosted-CPU competitive baseline measurement','meta':meta,'models':{},'claim_boundary':'Measured on a GitHub-hosted Ubuntu CPU runner, not the user target hardware. RSS is process MaxRSS from /usr/bin/time -v. No LARC comparison is claimed until an integrated LARC runtime is measured under the same protocol.'}
    for quant in ('Q4_K_M','Q2_K'):
        q={'file':meta['models'][quant], 'rss':{}, 'perplexity':parse_ppl(a.raw/f'ppl_{quant}.txt')}
        for mode in ('mmap','none'):
            q['rss'][mode]={}
            for ctx in (64,2048,8192):
                q['rss'][mode][str(ctx)]=parse_time(a.raw/f'time_{quant}_{ctx}_{mode}.txt')
        q['bench']=json.load(open(a.raw/f'bench_{quant}.json'))
        out['models'][quant]=q
    q4=out['models']['Q4_K_M'];q2=out['models']['Q2_K']
    out['comparisons']={
      'q2_file_size_ratio_vs_q4':q2['file']['bytes']/q4['file']['bytes'],
      'q4_file_reduction_vs_q2_x':q4['file']['bytes']/q2['file']['bytes'],
      'q2_ppl_ratio_vs_q4':q2['perplexity']['ppl']/q4['perplexity']['ppl'],
      'rss_q2_over_q4':{mode:{ctx:q2['rss'][mode][ctx]['max_rss_bytes']/q4['rss'][mode][ctx]['max_rss_bytes'] for ctx in ('64','2048','8192')} for mode in ('mmap','none')}
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
