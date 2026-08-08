#!/usr/bin/env python3
"""Run-6 partial external-model LARC conversion probe.

Loads a pretrained decoder-only Transformer, replaces ONE adjacent layer pair by
one physical layer object, function-prefits that object against the frozen
teacher's two layer calls, optionally performs hard-projected group-64 Q4 LM
recovery, then evaluates held-out NLL on disjoint text.

This is deliberately a partial conversion. It is meant to answer whether
cross-depth sharing can preserve a real pretrained model locally before a full
model conversion is attempted.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F


def texts():
    train=[
      "Once there was a small red kite that a child carried through the park on a windy afternoon.",
      "The engineer measured the system carefully before replacing the large component with a smaller equivalent.",
      "A rabbit found a blue box beside the garden and asked its friend to help open it.",
      "A good experiment keeps training examples separate from the final evaluation examples.",
      "The river moved slowly near the bank but became faster where the channel narrowed.",
      "A teacher explained the problem in several ways until every student could describe the reasoning.",
      "The computer reused a block of memory instead of allocating another identical copy.",
      "After the rain stopped the family walked home and watched the clouds disappear behind the hills.",
    ]
    cal=[f"Calibration example {i}. {train[i%len(train)]} {train[(i*3+1)%len(train)]}" for i in range(256)]
    ev=[f"Evaluation example {i}. {train[(i*5+2)%len(train)]} {train[(i*7+3)%len(train)]}" for i in range(128)]
    return cal,ev


def stream(tok,xs,n):
    out=[]
    for x in xs:
        out += tok.encode(x,add_special_tokens=False)
        if tok.eos_token_id is not None: out.append(tok.eos_token_id)
        if len(out)>=n+1:break
    seed=list(out)
    while len(out)<n+1:out.extend(seed)
    return torch.tensor(out[:n+1],dtype=torch.long)


def layers(model):
    for fn in (lambda m:m.model.layers,lambda m:m.model.model.layers,lambda m:m.transformer.h,lambda m:m.gpt_neox.layers):
        try:
            v=fn(model)
            if len(v):return v
        except Exception:pass
    raise RuntimeError("unsupported decoder layer layout")


def nll(model,ids,ctx,device,max_tokens):
    usable=(min(max_tokens,len(ids)-1)//ctx)*ctx;total=0.;count=0;model.eval()
    with torch.inference_mode():
        for s in range(0,usable,ctx):
            x=ids[s:s+ctx][None].to(device);y=ids[s+1:s+ctx+1][None].to(device);z=model(input_ids=x,use_cache=False).logits
            total+=float(F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction="sum"));count+=y.numel()
    return total/count,count


def average_state(a,b):
    sa,sb=a.state_dict(),b.state_dict();o={}
    for k,x in sa.items():
        y=sb[k]
        o[k]=((x.detach().float()+y.detach().float())*.5).to(x.dtype) if torch.is_floating_point(x) else x.detach().clone()
    return o


def q4_group64_(module):
    with torch.no_grad():
        seen=set()
        for p in module.parameters():
            ptr=p.untyped_storage().data_ptr()
            if ptr in seen:continue
            seen.add(ptr)
            if p.ndim<2:
                p.copy_(p.half().float());continue
            x=p.float();out=torch.empty_like(x)
            for s in range(0,x.shape[-1],64):
                a=x[...,s:s+64].reshape(-1,x[...,s:s+64].shape[-1]);pos=a.amax(-1).clamp_min(0)/7.;neg=(-a.amin(-1)).clamp_min(0)/8.;sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float();q=torch.round(a/sc[:,None]).clamp(-8,7)*sc[:,None];out[...,s:s+64].copy_(q.reshape_as(out[...,s:s+64]))
            p.copy_(out)


def make_batches(ids,ctx,batch,seed):
    g=torch.Generator().manual_seed(seed)
    def get():
        ix=torch.randint(0,len(ids)-ctx-1,(batch,),generator=g);x=torch.stack([ids[i:i+ctx] for i in ix]);y=torch.stack([ids[i+1:i+ctx+1] for i in ix]);return x,y
    return get


def forward_capture(model,which,input_ids,requires_grad):
    ls=layers(model);calls=[]
    def hook(_m,_inp,out):
        y=out[0] if isinstance(out,(tuple,list)) else out;calls.append(y if requires_grad else y.detach())
    hs=[ls[i].register_forward_hook(hook) for i in which]
    context=torch.enable_grad() if requires_grad else torch.inference_mode()
    with context: z=model(input_ids=input_ids,use_cache=False).logits
    for h in hs:h.remove()
    if len(calls)!=len(which):raise RuntimeError(f"expected {len(which)} layer calls, got {len(calls)}")
    return z,calls


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="HuggingFaceTB/SmolLM2-135M");ap.add_argument("--revision",default="d6a5589c239236d22370e2126bbe23d4843c47d9");ap.add_argument("--pair",default="auto");ap.add_argument("--context",type=int,default=128);ap.add_argument("--calibration-tokens",type=int,default=8192);ap.add_argument("--evaluation-tokens",type=int,default=8192);ap.add_argument("--prefit-steps",type=int,default=40);ap.add_argument("--recovery-steps",type=int,default=40);ap.add_argument("--batch",type=int,default=2);ap.add_argument("--lr",type=float,default=3e-4);ap.add_argument("--qat",action="store_true");ap.add_argument("--device",default="cpu");ap.add_argument("--output",type=Path,default=Path("benchmarks/run6_partial_layer_share.json"));a=ap.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    random.seed(606);torch.manual_seed(606);dev=torch.device(a.device);t0=time.time();cal_text,ev_text=texts();tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision);teacher=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.float32).to(dev).eval();student=copy.deepcopy(teacher).to(dev);tl,sl=layers(teacher),layers(student);n=len(tl)
    if a.pair=="auto":i=max(0,n//2-1);j=i+1
    else:i,j=map(int,a.pair.split(","))
    if j!=i+1:raise ValueError("Run-6 probe currently requires an adjacent pair")
    shared=copy.deepcopy(sl[i]);shared.load_state_dict(average_state(sl[i],sl[j]));sl[i]=shared;sl[j]=shared
    for p in student.parameters():p.requires_grad_(False)
    for p in shared.parameters():p.requires_grad_(True)
    cal=stream(tok,cal_text,a.calibration_tokens);ev=stream(tok,ev_text,a.evaluation_tokens);get=make_batches(cal,a.context,a.batch,6606)
    teacher_nll,N=nll(teacher,ev,a.context,dev,a.evaluation_tokens);initial_nll,_=nll(student,ev,a.context,dev,a.evaluation_tokens)
    opt=torch.optim.AdamW(shared.parameters(),lr=a.lr)
    curve=[]
    for step in range(a.prefit_steps):
        x,_=get();x=x.to(dev)
        _,to=forward_capture(teacher,[i,j],x,False);_,so=forward_capture(student,[i,j],x,True);loss=sum(F.mse_loss(s,t) for s,t in zip(so,to))/2;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(shared.parameters(),1.);opt.step();
        if a.qat:q4_group64_(shared)
        if (step+1)%10==0:curve.append({"phase":"prefit","step":step+1,"loss":float(loss.detach())});print(curve[-1],flush=True)
    post_prefit,_=nll(student,ev,a.context,dev,a.evaluation_tokens)
    opt=torch.optim.AdamW(shared.parameters(),lr=a.lr*.5)
    for step in range(a.recovery_steps):
        x,y=get();x=x.to(dev);y=y.to(dev);z=student(input_ids=x,use_cache=False).logits;loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(shared.parameters(),1.);opt.step();
        if a.qat:q4_group64_(shared)
        if (step+1)%10==0:curve.append({"phase":"lm_recovery","step":step+1,"loss":float(loss.detach())});print(curve[-1],flush=True)
    final,_=nll(student,ev,a.context,dev,a.evaluation_tokens);unique_before=sum(p.numel() for p in teacher.parameters());unique_after=sum(p.numel() for p in {id(p):p for p in student.parameters()}.values());out={"run":6,"evidence_level":"partial external-pretrained conversion probe","model":a.model,"revision":a.revision,"shared_pair":[i,j],"logical_layers":n,"context":a.context,"evaluation_tokens":N,"teacher_nll":teacher_nll,"student_initial_shared_nll":initial_nll,"student_post_prefit_nll":post_prefit,"student_final_nll":final,"final_delta_nats_per_token":final-teacher_nll,"final_perplexity_ratio":math.exp(final-teacher_nll),"prefit_steps":a.prefit_steps,"recovery_steps":a.recovery_steps,"group64_qat":a.qat,"unique_parameter_count_before":unique_before,"unique_parameter_count_after":unique_after,"unique_parameter_reduction_x":unique_before/unique_after,"curve":curve,"claim_boundary":"One adjacent-pair partial conversion only; this does not establish whole-model LARC compression, measured memory, or L3 completion.","wall_seconds":time.time()-t0};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
