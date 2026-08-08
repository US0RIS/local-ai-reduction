#!/usr/bin/env python3
from __future__ import annotations
import copy,json,math,random
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.run5_softshare_control import Teacher,D,H,HD,L,CHARS,train_teacher,evaluate,toks,batch,project_teacher_q4,project_soft_q4
from tools.run5_coreshare_control import CoreShare,core_bytes,teacher_bytes

def hidden_pairs(model,idx):
    B,T=idx.shape;x=model.emb(idx)+model.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1);ins=[];outs=[]
    with torch.inference_mode():
        for b in model.blocks:ins.append(x.detach().clone());x=b(x,mask);outs.append(x.detach().clone())
    return ins,outs

def layer_forward(m,x,l):
    B,T,_=x.shape;z=F.layer_norm(x,(D,),m.n1w[l],m.n1b[l]);q,k,v=m.lin(z,'qkv',l,m.qkv_bias[l]).chunk(3,-1);q=q.view(B,T,H,HD).transpose(1,2);k=k.view(B,T,H,HD).transpose(1,2);v=v.view(B,T,H,HD).transpose(1,2);a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D);x=x+m.lin(a,'o',l,m.o_bias[l]);z=F.layer_norm(x,(D,),m.n2w[l],m.n2b[l]);return x+m.lin(F.gelu(m.lin(z,'fc1',l,m.fc1_bias[l])),'fc2',l,m.fc2_bias[l])

def main():
    torch.manual_seed(3);random.seed(3);t=Teacher();train_teacher(t,120);tq=copy.deepcopy(t);project_teacher_q4(tq);ev=toks(333);teacher_nll,n=evaluate(tq,ev,32768)
    torch.manual_seed(123);xcal,_=batch(b=48,T=32);qin,qout=hidden_pairs(tq,xcal);ranks={'qkv':16,'o':16,'fc1':16,'fc2':16};torch.manual_seed(777);random.seed(777);s=CoreShare(tq,ranks);project_soft_q4(s);raw,_=evaluate(s,ev,32768)
    opt=torch.optim.AdamW(s.parameters(),lr=2e-4);rng=random.Random(100)
    for _ in range(400):
        l=rng.randrange(L);ids=torch.randint(0,xcal.shape[0],(8,));pred=layer_forward(s,qin[l][ids],l);loss=F.mse_loss(pred,qout[l][ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step();project_soft_q4(s)
    layer_nll,_=evaluate(s,ev,32768)
    opt=torch.optim.AdamW(s.parameters(),lr=1.5e-4)
    for _ in range(150):
        x,y=batch();z=s(x);loss=F.cross_entropy(z.reshape(-1,len(CHARS)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step();project_soft_q4(s)
    final,_=evaluate(s,ev,32768);reduction=teacher_bytes()/core_bytes(ranks)
    out={'run':5,'strategy':'CoreShare-10X streamed-teacher recovery','evidence_level':'controlled L2C recovery-feasibility study','source_teacher_representation':'canonical Q4_ROW','representation':'W_l = S + U C_l V^T, canonical Q4_ROW throughout recovery','model':{'hidden':D,'layers':L,'context':64,'rank':16},'training_seed':3,'evaluation_seed':333,'evaluation_chars':n,'calibration':{'sequences':48,'tokens_per_sequence':32,'source':'training corpus','teacher_hidden_pairs_cached':True},'teacher_q4_nll':teacher_nll,'raw_q4_source_coreshare_nll':raw,'after_400_layerwise_q4_teacher_mse_steps_nll':layer_nll,'after_layerwise_perplexity_ratio':math.exp(layer_nll-teacher_nll),'teacher_free_q4_ce_steps':150,'final_nll':final,'final_delta_nats_per_char':final-teacher_nll,'final_perplexity_ratio':math.exp(final-teacher_nll),'complete_toy_tensor_reduction_x':reduction,'teacher_required_during_final_ce_recovery':False,'claim_boundary':'This demonstrates a controlled recovery workflow compatible with streaming the source teacher to build layerwise calibration targets, then discarding teacher weights. It is not evidence that compression improves real-model intelligence and is not L3/L4.'}
    text=json.dumps(out,indent=2)+'\n';Path('benchmarks/run5_coreshare_streamed_recovery.json').write_text(text);print(text,end='')
if __name__=='__main__':main()
