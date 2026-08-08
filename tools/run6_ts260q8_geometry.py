#!/usr/bin/env python3
"""Run-6 geometry/sharing probe for committed TS260Q8 TinyStories checkpoints.

This path exists specifically to avoid Hugging Face/Xet. It can fetch the 18
row-Q8 model chunks committed by maddiedreese/paLLM, concatenate them, run a
faithful float reference of the quantized weights, and measure activation
geometry plus naive adjacent-layer reuse damage.

The resulting evidence is external-pretrained-model geometry evidence, not a
completed LARC conversion and not a measured-memory result.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.nn.functional as F

MAGIC=b"TS260Q8\0"
PALLM_COMMIT="ce6b82233bbff55a7876f09dcee35f5fa1f69535"
PALLM_TOTAL_BYTES=282_584
PALLM_CHUNKS=18

@dataclass(frozen=True)
class Config:
    dim:int; hidden_dim:int; n_layers:int; n_heads:int; n_kv_heads:int; vocab_size:int; max_seq_len:int

class TS260Q8Model:
    def __init__(self,path:Path):
        data=path.read_bytes();self.path=path;self.sha256=hashlib.sha256(data).hexdigest();off=0
        if data[:8]!=MAGIC:raise ValueError("not a TS260Q8 file")
        off+=8;self.cfg=Config(*struct.unpack_from("<7i",data,off));off+=28
        tensor_count,=struct.unpack_from("<I",data,off);off+=4;self.tensors={};self.tensor_shapes={}
        for _ in range(tensor_count):
            name_len,=struct.unpack_from("<H",data,off);off+=2;name=data[off:off+name_len].decode("utf-8");off+=name_len
            rows,cols=struct.unpack_from("<II",data,off);off+=8
            scales=torch.tensor(struct.unpack_from(f"<{rows}I",data,off),dtype=torch.float32)/65536.;off+=4*rows
            n=rows*cols;raw=torch.frombuffer(bytearray(data[off:off+n]),dtype=torch.uint8).clone().view(torch.int8).float().reshape(rows,cols);off+=n
            self.tensors[name]=raw*scales[:,None];self.tensor_shapes[name]=[rows,cols]
        if off!=len(data):raise ValueError(f"TS260Q8 parser left {len(data)-off} trailing bytes")
        c=self.cfg;D,Ff,L,H,K=c.dim,c.hidden_dim,c.n_layers,c.n_heads,c.n_kv_heads;kv_dim=D*K//H
        self.E=self.tensors["tok_embeddings"];self.rms_att=self.tensors["rms_att_weight"].reshape(L,D);self.rms_ffn=self.tensors["rms_ffn_weight"].reshape(L,D)
        self.wq=self.tensors["wq"].reshape(L,D,D);self.wk=self.tensors["wk"].reshape(L,kv_dim,D);self.wv=self.tensors["wv"].reshape(L,kv_dim,D);self.wo=self.tensors["wo"].reshape(L,D,D)
        self.w1=self.tensors["w1"].reshape(L,Ff,D);self.w2=self.tensors["w2"].reshape(L,D,Ff);self.w3=self.tensors["w3"].reshape(L,Ff,D)
        self.rms_final=self.tensors["rms_final_weight"][0];self.freq_real=self.tensors["freq_cis_real"];self.freq_imag=self.tensors["freq_cis_imag"];self.cls=self.tensors.get("wcls",self.E)
    @staticmethod
    def rmsnorm(x,w):return x*torch.rsqrt(x.square().mean()+1e-5)*w
    def forward_sequence(self,tokens,collector=None,override=None):
        c=self.cfg;D,H,L,K=c.dim,c.n_heads,c.n_layers,c.n_kv_heads;head_dim=D//H;kv_dim=D*K//H;kv_mul=H//K
        if len(tokens)>c.max_seq_len:raise ValueError("sequence exceeds checkpoint max_seq_len")
        key_cache=torch.zeros((L,c.max_seq_len,kv_dim));val_cache=torch.zeros_like(key_cache);logits=[]
        for pos,token in enumerate(tokens[:-1]):
            x=self.E[token].clone()
            for logical_layer in range(L):
                physical_layer=(override or {}).get(logical_layer,logical_layer);x_before=x.clone();z=self.rmsnorm(x,self.rms_att[physical_layer])
                if collector is not None:
                    for s in ("q_proj","k_proj","v_proj"):collector(logical_layer,s,z)
                q=self.wq[physical_layer]@z;k=self.wk[physical_layer]@z;v=self.wv[physical_layer]@z
                for i in range(0,D,2):
                    hi=i%head_dim;cr=float(self.freq_real[pos,hi//2]);ci=float(self.freq_imag[pos,hi//2]);q0,q1=float(q[i]),float(q[i+1]);q[i]=q0*cr-q1*ci;q[i+1]=q0*ci+q1*cr
                    if i<kv_dim:
                        k0,k1=float(k[i]),float(k[i+1]);k[i]=k0*cr-k1*ci;k[i+1]=k0*ci+k1*cr
                key_cache[logical_layer,pos]=k;val_cache[logical_layer,pos]=v;attn_value=torch.empty(D)
                for h in range(H):
                    qh=q[h*head_dim:(h+1)*head_dim];kvh=h//kv_mul;kh=key_cache[logical_layer,:pos+1,kvh*head_dim:(kvh+1)*head_dim];vh=val_cache[logical_layer,:pos+1,kvh*head_dim:(kvh+1)*head_dim]
                    probs=torch.softmax(kh@qh/math.sqrt(head_dim),dim=0);attn_value[h*head_dim:(h+1)*head_dim]=probs@vh
                if collector is not None:collector(logical_layer,"o_proj",attn_value)
                x=x+self.wo[physical_layer]@attn_value;z=self.rmsnorm(x,self.rms_ffn[physical_layer])
                if collector is not None:collector(logical_layer,"gate_proj",z);collector(logical_layer,"up_proj",z)
                hidden=F.silu(self.w1[physical_layer]@z)*(self.w3[physical_layer]@z)
                if collector is not None:collector(logical_layer,"down_proj",hidden)
                x=x+self.w2[physical_layer]@hidden
                if collector is not None:collector(logical_layer,"residual_delta",x-x_before)
            logits.append(self.cls@self.rmsnorm(x,self.rms_final))
        return torch.stack(logits)
    def generate(self,n_sequences,seq_len,seed,temperature=.85):
        if seq_len>self.cfg.max_seq_len:raise ValueError("requested sequence too long")
        g=torch.Generator().manual_seed(seed);out=[]
        for _ in range(n_sequences):
            seq=[1]
            while len(seq)<seq_len:
                p=torch.softmax(self.forward_sequence(seq+[0])[-1]/temperature,dim=0);seq.append(int(torch.multinomial(p,1,generator=g)))
            out.append(seq)
        return out

def fetch_pallm(output:Path):
    payload=bytearray()
    for i in range(PALLM_CHUNKS):
        url=f"https://raw.githubusercontent.com/maddiedreese/paLLM/{PALLM_COMMIT}/Src/q8_chunk_{i:03d}.bin";print("fetch",url,flush=True)
        with urllib.request.urlopen(url,timeout=60) as response:payload.extend(response.read())
    if len(payload)!=PALLM_TOTAL_BYTES:raise RuntimeError(f"expected {PALLM_TOTAL_BYTES} bytes, got {len(payload)}")
    output.write_bytes(payload);print("wrote",output,len(payload),hashlib.sha256(payload).hexdigest(),flush=True)

def collect_activations(model,sequences,cap=4096):
    buffers={}
    def add(layer,site,vector):
        b=buffers.setdefault((layer,site),[])
        if len(b)<cap:b.append(vector.detach().float().cpu())
    for seq in sequences:model.forward_sequence(seq,add)
    return {k:torch.stack(v) for k,v in buffers.items()}

def principal_basis(x,ranks):
    _,s,vh=torch.linalg.svd(x,full_matrices=False);total=x.square().sum().double().clamp_min(1e-30)
    energy={str(r):min(1.,float(s[:min(r,len(s))].double().square().sum()/total)) for r in ranks if r<=x.shape[1]};return vh,energy

def site_weight(m,l,s):return {"q_proj":m.wq[l],"k_proj":m.wk[l],"v_proj":m.wv[l],"o_proj":m.wo[l],"gate_proj":m.w1[l],"up_proj":m.w3[l],"down_proj":m.w2[l]}[s]

def output_nmse(weight,x,basis,ranks):
    y=x@weight.T;den=y.square().sum().double().clamp_min(1e-30);out={}
    for r in ranks:
        if r>x.shape[1]:continue
        br=basis[:min(r,len(basis))];xh=(x@br.T)@br;out[str(r)]=float((xh@weight.T-y).square().sum().double()/den)
    return out

def first_rank(values,predicate):
    for r in sorted(map(int,values)):
        if predicate(values[str(r)]):return r
    return None

def eval_nll(model,sequences,override=None):
    total=0.;count=0
    for seq in sequences:
        logits=model.forward_sequence(seq,override=override);targets=torch.tensor(seq[1:],dtype=torch.long);total+=float(F.cross_entropy(logits,targets,reduction="sum"));count+=len(targets)
    return total/count,count

def main():
    ap=argparse.ArgumentParser();ap.add_argument("model",type=Path,nargs="?",default=Path("pallm-ts260q8.bin"));ap.add_argument("--fetch-pallm",action="store_true");ap.add_argument("--output",type=Path,default=Path("benchmarks/run6_pallm_real_geometry.json"));ap.add_argument("--cal-seqs",type=int,default=16);ap.add_argument("--eval-seqs",type=int,default=16);ap.add_argument("--seq-len",type=int,default=48);ap.add_argument("--activation-cap",type=int,default=4096);a=ap.parse_args()
    if a.fetch_pallm:fetch_pallm(a.model)
    m=TS260Q8Model(a.model);ranks=[4,8,12,16,24,32,48,64,96,128];cal=m.generate(a.cal_seqs,a.seq_len,6101);ev=m.generate(a.eval_seqs,a.seq_len,6201);ca=collect_activations(m,cal,a.activation_cap);ea=collect_activations(m,ev,a.activation_cap);rows=[]
    for (l,s),xc in sorted(ca.items()):
        if s=="residual_delta":continue
        xe=ea[(l,s)];b,en=principal_basis(xc,ranks);no=output_nmse(site_weight(m,l,s),xe,b,ranks);rows.append({"layer":l,"site":s,"input_dim":xc.shape[1],"calibration_rows":len(xc),"evaluation_rows":len(xe),"activation_energy_fraction":en,"heldout_linear_output_nmse":no,"first_tested_rank_energy_ge_0_98":first_rank(en,lambda v:v>=.98),"first_tested_rank_energy_ge_0_99":first_rank(en,lambda v:v>=.99),"first_tested_rank_output_nmse_le_0_05":first_rank(no,lambda v:v<=.05),"first_tested_rank_output_nmse_le_0_02":first_rank(no,lambda v:v<=.02)});print(l,s,"E16",en.get("16"),"E32",en.get("32"),"NMSE16",no.get("16"),"NMSE32",no.get("32"),flush=True)
    depth=[]
    for l in range(m.cfg.n_layers-1):
        x,y=ca[(l,"residual_delta")],ca[(l+1,"residual_delta")];co=F.cosine_similarity(x,y,dim=-1);depth.append({"layers":[l,l+1],"mean_residual_delta_cosine":float(co.mean()),"median_residual_delta_cosine":float(co.median())})
    base,N=eval_nll(m,ev);reuse=[]
    for l in range(m.cfg.n_layers-1):
        q,_=eval_nll(m,ev,{l+1:l});d=q-base;reuse.append({"layers":[l,l+1],"mapping":f"logical layer {l+1} reuses physical layer {l}","baseline_nll":base,"shared_pair_nll":q,"delta_nats_per_token":d,"perplexity_ratio":math.exp(d)});print("reuse",l,l+1,"delta",d,flush=True)
    input_sites=[r for r in rows if r["input_dim"]==m.cfg.dim];summary={"fraction_dim64_sites_98pct_energy_by_rank16":sum(1 for r in input_sites if r["first_tested_rank_energy_ge_0_98"] is not None and r["first_tested_rank_energy_ge_0_98"]<=16)/max(1,len(input_sites)),"fraction_dim64_sites_output_nmse_le_0_05_by_rank16":sum(1 for r in input_sites if r["first_tested_rank_output_nmse_le_0_05"] is not None and r["first_tested_rank_output_nmse_le_0_05"]<=16)/max(1,len(input_sites)),"mean_naive_adjacent_reuse_delta_nats_per_token":sum(r["delta_nats_per_token"] for r in reuse)/max(1,len(reuse))}
    out={"run":6,"evidence_level":"external pretrained quantized-model geometry probe","source":"maddiedreese/paLLM committed TinyStories-260K row-Q8 payload","source_repository":"maddiedreese/paLLM","source_commit":PALLM_COMMIT,"source_payload_bytes":PALLM_TOTAL_BYTES,"model_sha256":m.sha256,"config":asdict(m.cfg),"tensor_shapes":m.tensor_shapes,"calibration_distribution":"deterministic model-generated token sequences, seed 6101","evaluation_distribution":"disjoint deterministic model-generated token sequences, seed 6201","calibration_sequences":a.cal_seqs,"evaluation_sequences":a.eval_seqs,"sequence_length":a.seq_len,"site_results":rows,"cross_depth_diagnostics":depth,"no_recovery_adjacent_reuse":reuse,"baseline_self_generated_nll":base,"evaluation_tokens":N,"summary":summary,"limitations":["checkpoint is already row-Q8 quantized rather than original FP32","evaluation token distribution is generated by the same model, not TinyStories validation text","260K-parameter model is much smaller than SmolLM2-135M","no LARC conversion or recovery is performed in this geometry probe"],"claim_boundary":"Real independently trained TinyStories-class model geometry/sharing evidence only. Not SmolLM2, not a completed L3 LARC conversion, and not measured process/device memory."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"model_sha256":m.sha256,"config":asdict(m.cfg),"baseline_nll":base,"summary":summary,"output":str(a.output)},indent=2),flush=True)
if __name__=="__main__":main()
