#include "larc_q4.h"
#include <cassert>

namespace larc {
static inline int q4_at(const std::uint8_t* row, std::size_t j) {
    const std::uint8_t b = row[j >> 1];
    const int code = (j & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
    return code - 8;
}

void q4_gemv(const Q4Rows& w, const float* x, float* y) {
    const std::size_t stride = (w.cols + 1) >> 1;
    for (std::size_t i = 0; i < w.rows; ++i) {
        const std::uint8_t* row = w.packed + i * stride;
        float acc = 0.0f;
        std::size_t j = 0;
        for (; j + 1 < w.cols; j += 2) {
            const std::uint8_t byte = row[j >> 1];
            const int q0 = int(byte & 0x0F) - 8;
            const int q1 = int((byte >> 4) & 0x0F) - 8;
            acc += float(q0) * x[j] + float(q1) * x[j + 1];
        }
        if (j < w.cols) acc += float(q4_at(row, j)) * x[j];
        y[i] = acc * w.scales[i];
    }
}

void q4_projected_gemv(const Q4Rows& b, const Q4Rows& a,
                       const float* x, float* rank_scratch, float* y) {
    assert(a.cols == b.rows);
    q4_gemv(b, x, rank_scratch);
    q4_gemv(a, rank_scratch, y);
}
} // namespace larc
