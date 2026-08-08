#!/usr/bin/env python3
"""Run-6 real-pretrained-model geometry/sharing falsification harness.

Measures activation-rank energy, held-out linear-output NMSE, adjacent-layer
residual-function similarity, and optional no-recovery pair-sharing NLL on an
external pretrained Transformer. Calibration and evaluation text are disjoint.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, statistics, time
from pathlib import Path
from typing import Any
import torch


def fallback_texts():
    base=[
      "Mathematics often replaces a complicated relationship with a simpler representation that preserves the quantities needed for prediction.",
      "A river transports sediment downstream while vegetation changes the speed and turbulence of water near the bank.",
      "Computer systems can be limited by memory bandwidth, arithmetic throughput, synchronization, or data movement depending on workload shape.",
      "A careful experiment separates calibration data from evaluation data so the final metric does not benefit from accidental leakage.",
      "Language models predict token sequences by repeatedly transforming a residual representation through attention and nonlinear layers.",
      "The committee compared several proposals, identified assumptions, tested alternatives, and recorded evidence before making a decision.",
      "A telescope gathers light over a large aperture and forms an image whose resolution depends on wavelength and instrument geometry.",
      "A compression method is useful only if quality, memory, runtime, and baseline are measured under compatible conditions.",
      "Scientists distinguish an observation, a model that explains it, and a prediction that can falsify the model.",
      "A legal agreement allocates rights, obligations, conditions, remedies, and risk among parties over a defined period.",
      "During training, an optimizer changes parameters in response to gradients computed from a finite sample of examples.",
      "The city expanded around rail lines and ports, then changed again as highways altered commuting patterns and land use."]
    cal=[f"Calibration {i}. {base[i%len(base)]} {base[(i*5+3)%len(base)]}" for i in range(320)]
    ev=[f"Evaluation {i}. {base[(i*7+1)%len(base)]} {base[(i*11+4)%len(base)]}" for i in range(160)]
    return cal,ev,"embedded_fallback_v1"


def load_texts():
    try:
        from datasets import load_dataset
        ds=load_dataset("wikitext","wikitext-2-raw-v1")
        cal=[x["text"] for x in ds["train"] if x["text"].strip()]
        ev=[x["text"] for x in ds["validation"] if x["text"].strip()]
        if len(cal)>=100 and len(ev)>=20:return cal,ev,"wikitext-2-raw-v1 train/validation"
    except Exception as exc: print("dataset fallback:",exc,flush=True)
    return fallback_texts()


def text_hash(xs):
    h=hashlib.sha256()
    for x in xs:h.update(x.encode(errors="replace")+b"\0")
    return h.hexdigest()


def token_stream(tok,texts,limit):
    out=[];eos=tok.eos_token_id
    for t in texts:
        out.extend(tok.encode(t,add_special_tokens=False))
        if eos is not None:out.append(eos)
        if len(out)>=limit+1:break
    if not out:raise RuntimeError("tokenizer produced no tokens")
    seed=list(out)
    while len(out)<limit+1:out.extend(seed)
    return torch.tensor(out[:limit+1],dtype=torch.long)


def get_layers(model):
    for name,fn in [("model.layers",lambda m:m.model.layers),("model.model.layers",lambda m:m.model.model.layers),("transformer.h",lambda m:m.transformer.h),("gpt_neox.layers",lambda m:m.gpt_neox.layers)]:
        try:
            x=fn(model)
            if len(x):return x,name
        except Exception:pass
    raise RuntimeError("unsupported decoder layer layout")


def choose_layers(spec,n):
    if spec=="all":return list(range(n))
    if spec=="auto":return sorted({i for i in [0,1,n//4,n//4+1,n//2,n//2+1,n-2,n-1] if 0<=i<n})
    x=sorted({int(i) for i in spec.split(",")})
    if not x or min(x)<0 or max(x)>=n:raise ValueError("invalid layer selection")
    return x


def sites(layer):
    paths={"q_proj":"self_attn.q_proj","k_proj":"self_attn.k_proj","v_proj":"self_attn.v_proj","o_proj":"self_attn.o_proj","gate_proj":"mlp.gate_proj","up_proj":"mlp.up_proj","down_proj":"mlp.down_proj"};out={}
    for name,path in paths.items():
        cur:Any=layer
        try:
            for p in path.split("."):cur=getattr(cur,p)
            if isinstance(cur,torch.nn.Linear):out[name]=cur
        except Exception:pass
    return out


class Rows:
    def __init__(self,cap):self.cap=cap;self.n=0;self.parts=[]
    def add(self,x):
        if self.n>=self.cap:return
        x=x.detach().reshape(-1,x.shape[-1]).float().cpu();need=self.cap-self.n
        if len(x)>need:x=x[torch.linspace(0,len(x)-1,need).round().long()]
        self.parts.append(x.half());self.n+=len(x)
    def tensor(self):
        if not self.parts:raise RuntimeError("empty activation collector")
        return torch.cat(self.parts).float()


def run(model,ids,ctx,device):
    usable=(len(ids)//ctx)*ctx;model.eval()
    with torch.inference_mode():
        for s in range(0,usable,ctx):model(input_ids=ids[s:s+ctx][None].to(device),use_cache=False)


def nll(model,ids,ctx,device,max_tokens):
    usable=(min(max_tokens,len(ids)-1)//ctx)*ctx;tot=0.;count=0;model.eval()
    with torch.inference_mode():
        for s in range(0,usable,ctx):
            x=ids[s:s+ctx][None].to(device);y=ids[s+1:s+ctx+1][None].to(device);z=model(input_ids=x,use_cache=False).logits
            tot+=float(torch.nn.functional.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction="sum"));count+=y.numel()
    return tot/count,count


def geometry(x,ranks,seed):
    q=min(max(ranks)+16,x.shape[0]-1,x.shape[1]);torch.manual_seed(seed)
    _,s,v=torch.pca_lowrank(x,q=q,center=False,niter=4);den=float(x.square().sum().double())
    e={str(r):float(s[:min(r,len(s))].double().square().sum()/max(den,1e-30)) for r in ranks}
    return v.t().contiguous(),e


def out_nmse(mod,x,basis,ranks):
    w=mod.weight.detach().float().cpu();b=None if mod.bias is None else mod.bias.detach().float().cpu();y=x@w.t();y=y if b is None else y+b;den=float(y.square().sum().double());out={}
    for r in ranks:
        br=basis[:min(r,len(basis))];yh=((x@br.t())@br)@w.t();yh=yh if b is None else yh+b
        out[str(r)]=float((yh-y).square().sum().double()/max(den,1e-30))
    return out


def first(vals,pred):
    for r in sorted(map(int,vals)):
        if pred(vals[str(r)]):return r
    return None


def state_distance(a,b):
    sa,sb=a.state_dict(),b.state_dict();num=den=0.
    for k in sa.keys()&sb.keys():
        x,y=sa[k],sb[k]
        if torch.is_floating_point(x) and x.shape==y.shape:
            xf=x.detach().float().cpu();yf=y.detach().float().cpu();num+=float((xf-yf).square().sum());den+=.5*float(xf.square().sum()+yf.square().sum())
    return math.sqrt(num/max(den,1e-30))


def avg_state(a,b):
    sa,sb=a.state_dict(),b.state_dict();out={}
    for k,x in sa.items():out[k]=((x.detach().float().cpu()+sb[k].detach().float().cpu())*.5).to(x.dtype) if torch.is_floating_point(x) else x.detach().cpu().clone()
    return out


def share_probe(model,layers,pairs,ids,ctx,device,tokens):
    base,n=nll(model,ids,ctx,device,tokens);rows=[]
    for i,j in pairs:
        si=copy.deepcopy(layers[i].state_dict());sj=copy.deepcopy(layers[j].state_dict());a=avg_state(layers[i],layers[j]);layers[i].load_state_dict(a);layers[j].load_state_dict(a)
        q,_=nll(model,ids,ctx,device,tokens);layers[i].load_state_dict(si);layers[j].load_state_dict(sj)
        rows.append({"layers":[i,j],"baseline_nll":base,"shared_pair_nll":q,"delta_nats_per_token":q-base,"perplexity_ratio":math.exp(q-base)});print("share",i,j,q-base,flush=True)
    return {"baseline_nll":base,"evaluation_tokens":n,"pairs":rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="HuggingFaceTB/SmolLM2-135M");ap.add_argument("--revision",default="d6a5589c239236d22370e2126bbe23d4843c47d9");ap.add_argument("--output",type=Path,default=Path("benchmarks/run6_real_activation_geometry.json"));ap.add_argument("--layers",default="auto");ap.add_argument("--ranks",default="8,16,32,48,64,96,128");ap.add_argument("--calibration-tokens",type=int,default=4096);ap.add_argument("--evaluation-tokens",type=int,default=4096);ap.add_argument("--sample-rows",type=int,default=1024);ap.add_argument("--context",type=int,default=256);ap.add_argument("--sharing-eval-tokens",type=int,default=2048);ap.add_argument("--skip-sharing-probe",action="store_true");ap.add_argument("--device",default="cpu");a=ap.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    random.seed(6006);torch.manual_seed(6006);dev=torch.device(a.device);t0=time.time();print("loading",a.model,a.revision,flush=True)
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision);model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.float32).to(dev).eval();layers,layer_path=get_layers(model);sel=choose_layers(a.layers,len(layers));ranks=sorted({int(x) for x in a.ranks.split(",") if int(x)>0})
    caltxt,evtxt,dataset=load_texts();calids=token_stream(tok,caltxt,a.calibration_tokens);evids=token_stream(tok,evtxt,max(a.evaluation_tokens,a.sharing_eval_tokens));print("selected",sel,"dataset",dataset,flush=True)
    cal={};ev={};delta={i:Rows(a.sample_rows) for i in sel};latest={};handles=[]
    def pre(store,key):
        def h(_m,inp):
            if inp and torch.is_tensor(inp[0]):store[key].add(inp[0])
        return h
    def dpre(i):
        def h(_m,inp):
            if inp and torch.is_tensor(inp[0]):latest[i]=inp[0].detach()
        return h
    def dpost(i):
        def h(_m,_inp,out):
            y=out[0] if isinstance(out,(tuple,list)) else out;x=latest.get(i)
            if x is not None and torch.is_tensor(y):delta[i].add(y.detach()-x)
        return h
    for i in sel:
        ss=sites(layers[i]);
        if not ss:raise RuntimeError(f"layer {i}: no Llama-family sites")
        for name,m in ss.items():cal[(i,name)]=Rows(a.sample_rows);handles.append(m.register_forward_pre_hook(pre(cal,(i,name))))
        handles+=[layers[i].register_forward_pre_hook(dpre(i)),layers[i].register_forward_hook(dpost(i))]
    run(model,calids[:-1],a.context,dev)
    for h in handles:h.remove()
    handles=[]
    for i in sel:
        for name,m in sites(layers[i]).items():ev[(i,name)]=Rows(a.sample_rows);handles.append(m.register_forward_pre_hook(pre(ev,(i,name))))
    run(model,evids[:a.evaluation_tokens],a.context,dev)
    for h in handles:h.remove()
    rows=[]
    for i,name in sorted(cal):
        xc=cal[(i,name)].tensor();xe=ev[(i,name)].tensor();m=sites(layers[i])[name];basis,en=geometry(xc,ranks,6006+i*101+sum(map(ord,name)));nm=out_nmse(m,xe,basis,ranks)
        row={"layer":i,"site":name,"input_dim":xc.shape[1],"calibration_rows":len(xc),"evaluation_rows":len(xe),"uncentered_activation_energy_fraction":en,"heldout_linear_output_nmse":nm,"first_tested_rank_energy_ge_0_98":first(en,lambda v:v>=.98),"first_tested_rank_energy_ge_0_99":first(en,lambda v:v>=.99),"first_tested_rank_output_nmse_le_0_05":first(nm,lambda v:v<=.05),"first_tested_rank_output_nmse_le_0_02":first(nm,lambda v:v<=.02)};rows.append(row);print(i,name,"E32",en.get("32"),"NMSE32",nm.get("32"),flush=True)
    adj=[(i,i+1) for i in sel if i+1 in sel];depth=[]
    for i,j in adj:
        x,y=delta[i].tensor(),delta[j].tensor();n=min(len(x),len(y));x=torch.nn.functional.normalize(x[:n],dim=-1);y=torch.nn.functional.normalize(y[:n],dim=-1);c=(x*y).sum(-1);depth.append({"layers":[i,j],"mean_residual_delta_cosine":float(c.mean()),"median_residual_delta_cosine":float(c.median()),"rms_relative_state_distance":state_distance(layers[i],layers[j]),"rows":n})
    sharing=None if a.skip_sharing_probe else share_probe(model,layers,adj,evids,a.context,dev,a.sharing_eval_tokens)
    cfg=model.config.to_dict();summary={"fraction_sites_energy_98pct_by_rank64":statistics.mean([1. if x["first_tested_rank_energy_ge_0_98"] is not None and x["first_tested_rank_energy_ge_0_98"]<=64 else 0. for x in rows]),"fraction_sites_output_nmse_le_0_05_by_rank64":statistics.mean([1. if x["first_tested_rank_output_nmse_le_0_05"] is not None and x["first_tested_rank_output_nmse_le_0_05"]<=64 else 0. for x in rows])}
    out={"run":6,"evidence_level":"real-pretrained geometry probe; not a converted L3 model","model":a.model,"revision":a.revision,"model_type":cfg.get("model_type"),"hidden_size":cfg.get("hidden_size") or cfg.get("n_embd"),"intermediate_size":cfg.get("intermediate_size"),"num_hidden_layers":cfg.get("num_hidden_layers") or cfg.get("n_layer"),"num_attention_heads":cfg.get("num_attention_heads") or cfg.get("n_head"),"num_key_value_heads":cfg.get("num_key_value_heads"),"layer_path":layer_path,"selected_layers":sel,"candidate_ranks":ranks,"dataset":dataset,"calibration_text_sha256":text_hash(caltxt),"evaluation_text_sha256":text_hash(evtxt),"calibration_tokens":a.calibration_tokens,"evaluation_tokens":a.evaluation_tokens,"sample_rows_per_site":a.sample_rows,"context":a.context,"site_results":rows,"cross_depth_diagnostics":depth,"no_recovery_pair_sharing_probe":sharing,"summary":summary,"claim_boundary":"Real pretrained activation geometry and naive pair-sharing probe only; no compressed-model, measured-memory, or retained-intelligence claim.","wall_seconds":time.time()-t0};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"summary":summary,"output":str(a.output)},indent=2),flush=True)

if __name__=="__main__":main()
