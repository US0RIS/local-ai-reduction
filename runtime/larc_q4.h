#pragma once
#include <cstddef>
#include <cstdint>

namespace larc {
struct Q4Rows {
    const std::uint8_t* packed;
    const float* scales;
    std::size_t rows;
    std::size_t cols;
};
void q4_gemv(const Q4Rows& w, const float* x, float* y);
void q4_projected_gemv(const Q4Rows& b, const Q4Rows& a,
                       const float* x, float* rank_scratch, float* y);
inline std::size_t projected_scratch_bytes(const Q4Rows& b) {
    return b.rows * sizeof(float);
}
} // namespace larc
