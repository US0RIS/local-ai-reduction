#include "larc_q2_attention.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace larc {
float e4m3fn_to_float(std::uint8_t b) {
    const int sign=(b&0x80u)?-1:1;
    const int exp=(b>>3)&0x0F;
    const int mant=b&0x07;
    if(exp==0) {
        if(mant==0) return sign*0.0f;
        return sign*std::ldexp(float(mant)/8.0f,-6);
    }
    // E4M3-FN reserves only exponent=15,mantissa=7 for NaN; finite max is 448.
    if(exp==15 && mant==7) return std::numeric_limits<float>::quiet_NaN();
    return sign*std::ldexp(1.0f+float(mant)/8.0f,exp-7);
}

static inline int q2_at(const std::uint8_t* row,std::size_t j) {
    return int((row[j>>2]>>((j&3)*2))&0x03u);
}

static void metric_mv(const std::uint16_t* metric,std::size_t rank,
                      const float* x,float* y) {
    for(std::size_t i=0;i<rank;++i) {
        float acc=0.0f;
        for(std::size_t j=0;j<rank;++j)
            acc+=fp16_bits_to_float(metric[i*rank+j])*x[j];
        y[i]=acc;
    }
}

void latent_q2_fp8_attention_head(
    const Q4Rows& kb,const std::uint16_t* k_metric_fp16,
    const Q4Rows& vb,const std::uint16_t* v_metric_fp16,
    const Q2RowsFP8& keys,const Q2RowsFP8& values,
    const float* query,float inv_sqrt_head_dim,
    float* scratch,float* output) {
    const std::size_t rank=kb.rows;
    const std::size_t tokens=keys.rows;
    if(vb.rows!=rank || keys.rank!=rank || values.rank!=rank || values.rows!=tokens)
        return;

    float* q_lat=scratch;
    float* q_metric=q_lat+rank;
    float* scores=q_metric+rank;
    float* v_acc=scores+tokens;
    float* v_corrected=v_acc+rank;

    q4_gemv(kb,query,q_lat);
    metric_mv(k_metric_fp16,rank,q_lat,q_metric);

    const std::size_t stride=(rank+3)>>2;
    float max_score=-std::numeric_limits<float>::infinity();
    for(std::size_t t=0;t<tokens;++t) {
        const float mn=e4m3fn_to_float(keys.min_e4m3fn[t]);
        const float scale=e4m3fn_to_float(keys.scale_e4m3fn[t]);
        const std::uint8_t* row=keys.packed+t*stride;
        float score=0.0f;
        for(std::size_t j=0;j<rank;++j)
            score+=q_metric[j]*(mn+scale*float(q2_at(row,j)));
        score*=inv_sqrt_head_dim;
        scores[t]=score;
        max_score=std::max(max_score,score);
    }

    float denominator=0.0f;
    for(std::size_t t=0;t<tokens;++t) {
        scores[t]=std::exp(scores[t]-max_score);
        denominator+=scores[t];
    }

    std::fill(v_acc,v_acc+rank,0.0f);
    for(std::size_t t=0;t<tokens;++t) {
        const float alpha=scores[t]/denominator;
        const float mn=e4m3fn_to_float(values.min_e4m3fn[t]);
        const float scale=e4m3fn_to_float(values.scale_e4m3fn[t]);
        const std::uint8_t* row=values.packed+t*stride;
        for(std::size_t j=0;j<rank;++j)
            v_acc[j]+=alpha*(mn+scale*float(q2_at(row,j)));
    }

    metric_mv(v_metric_fp16,rank,v_acc,v_corrected);
    q4_transposed_gemv(vb,v_corrected,output);
}
} // namespace larc
