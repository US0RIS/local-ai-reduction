#!/usr/bin/env python3
import json,math
from pathlib import Path

HIDDEN=576;FF=1536;VOCAB=49152;LAYERS=30
PROJECTIONS=[('q_proj',576,576),('k_proj',192,576),('v_proj',192,576),('o_proj',576,576),('gate_proj',1536,576),('up_proj',1536,576),('down_proj',576,1536)]

def row_q4(rows,cols):return rows*((cols+1)//2+2)
def group64_q4(rows,cols):return rows*((cols+1)//2+math.ceil(cols/64)*2)

def main():
    parts=[];rb=gb=0
    for name,rows,cols in PROJECTIONS:
        r=row_q4(rows,cols);g=group64_q4(rows,cols);parts.append({'name':name,'rows':rows,'cols':cols,'row_q4_bytes':r,'group64_q4_bytes':g});rb+=r;gb+=g
    norms=2*HIDDEN*2;rb+=norms;gb+=norms
    embed=row_q4(VOCAB,HIDDEN);baseline=embed+LAYERS*rb+HIDDEN*2
    group_layers=4;student=baseline-group_layers*rb+gb
    out={'model':'SmolLM2-135M geometry','source_geometry':{'hidden':HIDDEN,'ff':FF,'vocab':VOCAB,'layers':LAYERS,'q_heads':9,'kv_heads':3,'head_dim':64},'per_layer':{'projection_breakdown':parts,'rmsnorm_fp16_bytes':norms,'row_q4_total_bytes':rb,'group64_q4_total_bytes':gb},'partial_4_to_1':{'baseline_four_row_q4_blocks_bytes':4*rb,'shared_one_group64_block_bytes':gb,'group_weight_reduction_x':4*rb/gb,'whole_model_baseline_row_q4_weight_bytes':baseline,'whole_model_partial_shared_weight_bytes':student,'whole_model_weight_reduction_x':baseline/student,'whole_model_weight_bytes_saved_fraction':(baseline-student)/baseline},'note':'Structural arithmetic only. Embedding is counted once because LM head is tied. No KV or runtime scratch is included.'}
    Path('benchmarks/run6_partial_memory_plan.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
