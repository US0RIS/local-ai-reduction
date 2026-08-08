#pragma once
#include <cstddef>
#include <cstdint>

namespace larc {
// Canonical Q4_ROW storage contract:
// - decoded integer = nibble code - 8, range [-8,7]
// - two codes per byte, low nibble first
// - IEEE-754 binary16 scales stored as raw uint16 bits
struct Q4Rows {
    const std::uint8_t* packed;
    const std::uint16_t* scales_fp16;
    std::size_t rows;
    std::size_t cols;
};

// Same nibble contract, but one scale per contiguous column group in each row.
// Run-5 controlled profile uses group_size=64. scales_fp16 is row-major with
// ceil(cols/group_size) scale entries per row.
struct Q4GroupRows {
    const std::uint8_t* packed;
    const std::uint16_t* scales_fp16;
    std::size_t rows;
    std::size_t cols;
    std::size_t group_size;
};

float fp16_bits_to_float(std::uint16_t h);
std::uint16_t float_to_fp16_bits(float f);

void q4_gemv(const Q4Rows& w,const float* x,float* y);
void q4_transposed_gemv(const Q4Rows& w,const float* x,float* y);
void q4_projected_gemv(const Q4Rows& b,const Q4Rows& a,
                       const float* x,float* rank_scratch,float* y);
void q4_grouped_gemv(const Q4GroupRows& w,const float* x,float* y);

inline std::size_t projected_scratch_bytes(const Q4Rows& b) {
    return b.rows*sizeof(float);
}
inline std::size_t q4_storage_bytes(const Q4Rows& w) {
    return w.rows*((w.cols+1)/2)+w.rows*sizeof(std::uint16_t);
}
inline std::size_t q4_grouped_storage_bytes(const Q4GroupRows& w) {
    const std::size_t groups=(w.cols+w.group_size-1)/w.group_size;
    return w.rows*((w.cols+1)/2)+w.rows*groups*sizeof(std::uint16_t);
}
} // namespace larc
