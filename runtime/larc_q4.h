#pragma once
#include <cstddef>
#include <cstdint>

namespace larc {
struct Q4Rows {
    const std::uint8_t* packed;
    const std::uint16_t* scales_fp16;
    std::size_t rows;
    std::size_t cols;
};

float fp16_bits_to_float(std::uint16_t h);
std::uint16_t float_to_fp16_bits(float f);
void q4_gemv(const Q4Rows& w,const float* x,float* y);
void q4_gemv_add(const Q4Rows& w,const float* x,float* y);
void q4_transposed_gemv(const Q4Rows& w,const float* x,float* y);
void q4_projected_gemv(const Q4Rows& b,const Q4Rows& a,const float* x,float* rank_scratch,float* y);
// SoftShare: y = Sx + A(Bx).
void q4_shared_residual_gemv(const Q4Rows& shared,const Q4Rows& b,const Q4Rows& a,const float* x,float* rank_scratch,float* y);
// CoreShare: y = Sx + U(C(V^T x)). vt is stored directly as [rank,input].
// scratch_a and scratch_b are each rank floats. No per-layer dense W is materialized.
void q4_shared_core_gemv(const Q4Rows& shared,const Q4Rows& vt,const Q4Rows& core,const Q4Rows& u,
                         const float* x,float* scratch_a,float* scratch_b,float* y);
inline std::size_t projected_scratch_bytes(const Q4Rows& b) { return b.rows*sizeof(float); }
inline std::size_t coreshare_scratch_bytes(const Q4Rows& vt) { return 2*vt.rows*sizeof(float); }
inline std::size_t q4_storage_bytes(const Q4Rows& w) { return w.rows*((w.cols+1)/2)+w.rows*sizeof(std::uint16_t); }
} // namespace larc
