#pragma once
#include "larc_q4.h"
#include <cstddef>
#include <cstdint>

namespace larc {
struct Q2RowsFP8 {
    const std::uint8_t* packed;
    const std::uint8_t* min_e4m3fn;
    const std::uint8_t* scale_e4m3fn;
    std::size_t rows;
    std::size_t rank;
};

float e4m3fn_to_float(std::uint8_t b);

// One-head autoregressive attention scratch: two query/rank vectors, T scores,
// one value accumulator and one metric-corrected value vector. Historical
// latent K/V are consumed directly from packed Q2 and are never materialized
// as an FP32 T x rank matrix.
inline std::size_t q2_attention_scratch_floats(std::size_t tokens,std::size_t rank) {
    return tokens+4*rank;
}

void latent_q2_fp8_attention_head(
    const Q4Rows& k_basis,const std::uint16_t* k_metric_fp16,
    const Q4Rows& v_basis,const std::uint16_t* v_metric_fp16,
    const Q2RowsFP8& keys,const Q2RowsFP8& values,
    const float* query,float inv_sqrt_head_dim,
    float* scratch,float* output);
} // namespace larc
