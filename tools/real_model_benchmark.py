#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, math, os, tempfile, time, zlib
from pathlib import Path
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from larc.q4_runtime import Q4Basis, Q4ProjectedLinear, Q4VocabFactors, LARCEmbedding, LARCHead, fit_basis

MODEL_ID='HuggingFaceTB/SmolLM2-135M'
# Exact published Q4_K_M SmolLM2-135M GGUF size is reported as 105 MB by
# multiple Hugging Face conversions. Keep the baseline named in every result.
GGUF_Q4_K_M_BYTES=105_000_000
PROFILES={
 '10x': dict(hidden=32,o=32,down=48,vocab=128),
 '15x': dict(hidden=20,o=20,down=32,vocab=80),
 '20x': dict(hidden=12,o=12,down=20,vocab=48),
 '30x': dict(hidden=8,o=8,down=12,vocab=32),
}
CALIBRATION=[
"A scientist carefully records the temperature of a metal sample while it cools, then compares the measurements with a simple physical model.",
"The city council approved a plan to repair the bridge after engineers found corrosion in several supporting beams.",
"When the child opened the old wooden box, she found a collection of letters, photographs, and a small silver key.",
"A computer program receives a list of numbers, removes duplicates, sorts the remaining values, and prints the median.",
"The history of navigation changed when sailors learned to combine astronomical observations with accurate clocks and detailed charts.",
"In a healthy ecosystem, energy moves through plants, herbivores, predators, and decomposers while nutrients are repeatedly recycled.",
"The lawyer reviewed the agreement section by section, checking the closing conditions, representations, covenants, and indemnification provisions.",
"A baker mixes flour, water, yeast, and salt, allows the dough to ferment, then shapes and bakes the loaf until the crust is browned.",
"The spacecraft adjusted its trajectory with a short engine burn so that it would pass the planet at the planned altitude.",
"Machine learning systems can memorize patterns in training data, so evaluation should use examples that were not used during optimization.",
]*6
EVAL_TEXTS=[
"The rain began just after sunrise. By noon the streets were shining, and people hurried beneath umbrellas while buses moved slowly through traffic. In the afternoon the clouds broke apart and a narrow band of blue appeared over the western hills.",
"To solve a difficult problem, it often helps to divide it into smaller parts. Each part can be checked independently, and the intermediate results can reveal whether an early assumption was wrong before the entire calculation is complete.",
"Electric current is the rate at which charge passes a point in a circuit. A resistor opposes that motion, and the relationship among voltage, current, and resistance can be described by Ohm's law.",
"Maria placed three books on the desk and gave one to Daniel. Two books remained on the desk. Daniel thanked her and began reading the first chapter while Maria returned the other books to the shelf.",
"A contract can allocate risk in several ways. Price adjustments address changes in value, representations describe facts at signing or closing, covenants govern conduct, and indemnities can assign responsibility for specified losses.",
"The small robot moved forward until its distance sensor detected a wall. It stopped, rotated ninety degrees, checked the new path, and continued down the corridor without touching the obstacle.",
"Photosynthesis allows plants to use light energy to convert carbon dioxide and water into chemical energy. Oxygen is released as a byproduct, and the stored energy later supports growth and metabolism.",
"Good software tests include ordinary cases, boundary cases, and intentionally invalid inputs. A test suite is most useful when failures identify a specific broken assumption rather than merely reporting that something went wrong.",
]
GEN_PROMPTS=["Once upon a time, a curious girl found a tiny door behind a bookshelf.","The most important reason to test a scientific hypothesis is"]

def rss_mb(): return psutil.Process(os.getpid()).memory_info().rss/1e6

def token_batch(tok,texts,max_length=512):
    return tok("\n\n".join(texts),return_tensors='pt',truncation=True,max_length=max_length,add_special_tokens=True)['input_ids']

def eval_nll(model,tok):
    model.eval(); ids=token_batch(tok,EVAL_TEXTS,512)
    with torch.inference_mode():
        out=model(ids,use_cache=False)
        logits=out.logits[:,:-1].float(); target=ids[:,1:]
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1)).item()
    return loss, math.exp(min(loss,20)), int(target.numel())

def capture_activations(model,tok,max_tokens=384):
    caps={}; handles=[]
    def hook(name):
      def _h(mod,args):
        x=args[0].detach().reshape(-1,args[0].shape[-1]).cpu()
        if x.shape[0]>max_tokens: x=x[:max_tokens]
        caps[name]=x.to(torch.float16)
      return _h
    for i,layer in enumerate(model.model.layers):
      for tag,mod in [('attn',layer.self_attn.q_proj),('o',layer.self_attn.o_proj),('mlp',layer.mlp.gate_proj),('down',layer.mlp.down_proj)]:
        handles.append(mod.register_forward_pre_hook(hook(f'{i}.{tag}')))
    ids=token_batch(tok,CALIBRATION,max_tokens)
    with torch.inference_mode(): model(ids,use_cache=False)
    for h in handles: h.remove()
    return caps

def factorize_vocab(model,rank):
    E=model.model.embed_tokens.weight.detach().float().cpu()
    q=min(rank+12,min(E.shape))
    # Low-rank randomized SVD of tied embedding/head matrix.
    _,_,V=torch.pca_lowrank(E,q=q,center=False,niter=3)
    V=V[:,:rank].contiguous(); C=E@V
    factors=Q4VocabFactors(C,V)
    model.model.embed_tokens=LARCEmbedding(factors)
    model.lm_head=LARCHead(factors)
    return factors

def factorize_model(model,caps,p):
    for i,layer in enumerate(model.model.layers):
      # q/k/v and gate/up consume the same residual-stream width. A single
      # calibration-derived basis is shared across these five logical tensors.
      xh=torch.cat([caps[f'{i}.attn'].float(),caps[f'{i}.mlp'].float()],dim=0)
      bh=fit_basis(xh,p['hidden']); qbh=Q4Basis(bh)
      for modname in ['q_proj','k_proj','v_proj']:
        old=getattr(layer.self_attn,modname); A=old.weight.detach().float().cpu()@bh.t(); setattr(layer.self_attn,modname,Q4ProjectedLinear(qbh,A,old.bias))
      for modname in ['gate_proj','up_proj']:
        old=getattr(layer.mlp,modname); A=old.weight.detach().float().cpu()@bh.t(); setattr(layer.mlp,modname,Q4ProjectedLinear(qbh,A,old.bias))
      bo=fit_basis(caps[f'{i}.o'].float(),p['o']); qbo=Q4Basis(bo); old=layer.self_attn.o_proj; layer.self_attn.o_proj=Q4ProjectedLinear(qbo,old.weight.detach().float().cpu()@bo.t(),old.bias)
      bd=fit_basis(caps[f'{i}.down'].float(),p['down']); qbd=Q4Basis(bd); old=layer.mlp.down_proj; layer.mlp.down_proj=Q4ProjectedLinear(qbd,old.weight.detach().float().cpu()@bd.t(),old.bias)
      if (i+1)%5==0: print('factorized layer',i+1,flush=True)

def unique_model_bytes(model):
    seen=set(); total=0
    for t in list(model.parameters())+list(model.buffers()):
      if t.device.type=='meta': continue
      ptr=t.untyped_storage().data_ptr() if t.numel() else id(t)
      if ptr in seen: continue
      seen.add(ptr); total += t.untyped_storage().nbytes()
    return total

def compressed_tokenizer_bytes(tok):
    with tempfile.TemporaryDirectory() as td:
      files=tok.save_pretrained(td); raw=b''
      for f in files:
        if os.path.isfile(f): raw += Path(f).read_bytes()
      return len(zlib.compress(raw,9)),len(raw)

def generate_no_cache(model,tok,prompt,new_tokens=12):
    ids=tok(prompt,return_tensors='pt')['input_ids']
    with torch.inference_mode():
      for _ in range(new_tokens):
        logits=model(ids,use_cache=False).logits[:,-1]
        nxt=logits.argmax(-1,keepdim=True); ids=torch.cat([ids,nxt],dim=1)
    return tok.decode(ids[0],skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--profile',choices=PROFILES,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); p=PROFILES[args.profile]
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 2))); torch.manual_seed(0)
    print('loading',MODEL_ID,flush=True); t0=time.time(); tok=AutoTokenizer.from_pretrained(MODEL_ID); model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float32,low_cpu_mem_usage=True); load_s=time.time()-t0
    base_loss,base_ppl,ntok=eval_nll(model,tok); print('baseline',base_loss,base_ppl,'rss',rss_mb(),flush=True)
    caps=capture_activations(model,tok); print('captured',len(caps),'rss',rss_mb(),flush=True)
    t0=time.time(); factorize_vocab(model,p['vocab']); factorize_model(model,caps,p); del caps; gc.collect(); compress_s=time.time()-t0
    encoded=unique_model_bytes(model); tok_z,tok_raw=compressed_tokenizer_bytes(tok); larc_file=encoded+tok_z+65536
    ratio=GGUF_Q4_K_M_BYTES/larc_file
    loss,ppl,_=eval_nll(model,tok); print('compressed',loss,ppl,'bytes',encoded,'file',larc_file,'ratio',ratio,'rss',rss_mb(),flush=True)
    ids=token_batch(tok,EVAL_TEXTS,128); t0=time.time()
    with torch.inference_mode(): model(ids,use_cache=False)
    fwd_s=time.time()-t0
    samples=[generate_no_cache(model,tok,gp,12) for gp in GEN_PROMPTS]
    result={
      'model':MODEL_ID,'profile':args.profile,'ranks':p,
      'q4_gguf_reference_bytes':GGUF_Q4_K_M_BYTES,
      'q4_gguf_reference':'SmolLM2-135M Q4_K_M published Hugging Face conversion, 105 MB',
      'larc_resident_weight_bytes':encoded,'tokenizer_raw_bytes':tok_raw,'tokenizer_zlib_bytes':tok_z,'larc_estimated_file_bytes':larc_file,
      'compression_multiple_vs_q4_gguf':ratio,'baseline_eval_nll':base_loss,'baseline_eval_ppl':base_ppl,'larc_eval_nll':loss,'larc_eval_ppl':ppl,
      'nll_increase_pct':(loss/base_loss-1)*100,'ppl_multiple':ppl/base_ppl,'eval_tokens':ntok,'forward_128_seconds':fwd_s,
      'load_seconds':load_s,'compression_seconds':compress_s,'post_compression_process_rss_mb':rss_mb(),'samples':samples,
      'passes_file_target': ratio>=float(args.profile[:-1]),
      'passes_reasonable_quality_gate': loss <= base_loss*1.15,
      'notes':'Run-2 screening quality gate is <=15% held-out NLL increase. Passing is necessary but not sufficient for a comparable-intelligence claim.'
    }
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
