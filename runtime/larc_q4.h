#pragma once
#include <cstddef>
#include <cstdint>

namespace larc {
// Canonical Q4_ROW storage contract:
// - two's-offset nibble code: decoded integer = code - 8, range [-8,7]
// - two codes per byte, low nibble first
// - one IEEE-754 binary16 scale per row, stored as raw uint16 bits
struct Q4Rows {
    const std::uint8_t* packed;
    const std::uint16_t* scales_fp16;
    std::size_t rows;
    std::size_t cols;
};

float fp16_bits_to_float(std::uint16_t h);
std::uint16_t float_to_fp16_bits(float f);

void q4_gemv(const Q4Rows& w,const float* x,float* y);
// y = W^T x without storing a separate transposed basis.
void q4_transposed_gemv(const Q4Rows& w,const float* x,float* y);
void q4_projected_gemv(const Q4Rows& b,const Q4Rows& a,
                       const float* x,float* rank_scratch,float* y);
inline std::size_t projected_scratch_bytes(const Q4Rows& b) {
    return b.rows*sizeof(float);
}
inline std::size_t q4_storage_bytes(const Q4Rows& w) {
    return w.rows*((w.cols+1)/2)+w.rows*sizeof(std::uint16_t);
}
} // namespace larc
