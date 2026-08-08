#include "larc_q2_attention.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace larc {
float e4m3fn_to_float(std::uint8_t b) {
    const int sign=(b&0x80u)?-1:1;const int exp=(b>>3)&0x0F;const int mant=b&0x07;
    if(exp==0){if(mant==0)return sign*0.0f;return sign*std::ldexp(float(mant)/8.0f,-6);}
    if(exp==15 && mant==7)return std::numeric_limits<float>::quiet_NaN();
    return sign*std::ldexp(1.0f+float(mant)/8.0f,exp-7);
}
static inline int q2_at(const std::uint8_t* row,std::size_t j){return int((row[j>>2]>>((j&3)*2))&0x03u);}
static void metric_mv(const std::uint16_t* metric,std::size_t rank,const float* x,float* y){for(std::size_t i=0;i<rank;++i){float acc=0.0f;for(std::size_t j=0;j<rank;++j)acc+=fp16_bits_to_float(metric[i*rank+j])*x[j];y[i]=acc;}}
void latent_q2_fp8_attention_head(const Q4Rows& kb,const std::uint16_t* km,const Q4Rows& vb,const std::uint16_t* vm,const Q2RowsFP8& keys,const Q2RowsFP8& values,const float* query,float inv_sqrt_head_dim,float* scratch,float* output){
    const std::size_t rank=kb.rows,tokens=keys.rows;if(vb.rows!=rank||keys.rank!=rank||values.rank!=rank||values.rows!=tokens)return;
    float* qlat=scratch;float* qmetric=qlat+rank;float* scores=qmetric+rank;float* vacc=scores+tokens;float* vc=vacc+rank;
    q4_gemv(kb,query,qlat);metric_mv(km,rank,qlat,qmetric);const std::size_t stride=(rank+3)>>2;float maxs=-std::numeric_limits<float>::infinity();
    for(std::size_t t=0;t<tokens;++t){float mn=e4m3fn_to_float(keys.min_e4m3fn[t]),sc=e4m3fn_to_float(keys.scale_e4m3fn[t]);const auto* row=keys.packed+t*stride;float s=0;for(std::size_t j=0;j<rank;++j)s+=qmetric[j]*(mn+sc*float(q2_at(row,j)));s*=inv_sqrt_head_dim;scores[t]=s;maxs=std::max(maxs,s);}
    float den=0;for(std::size_t t=0;t<tokens;++t){scores[t]=std::exp(scores[t]-maxs);den+=scores[t];}
    std::fill(vacc,vacc+rank,0.0f);for(std::size_t t=0;t<tokens;++t){float a=scores[t]/den,mn=e4m3fn_to_float(values.min_e4m3fn[t]),sc=e4m3fn_to_float(values.scale_e4m3fn[t]);const auto* row=values.packed+t*stride;for(std::size_t j=0;j<rank;++j)vacc[j]+=a*(mn+sc*float(q2_at(row,j)));}
    metric_mv(vm,rank,vacc,vc);q4_transposed_gemv(vb,vc,output);
}
} // namespace larc
