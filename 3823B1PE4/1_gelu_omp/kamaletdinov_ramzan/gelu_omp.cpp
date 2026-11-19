#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    constexpr float k2 = 2.0f * 0.7978845608028654f;
    constexpr float k3 = k2 * 0.044715f;

    const float* in = input.data();
    float* out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const float x = in[i];
        const float t = x * (k2 + k3 * x * x);
        out[i] = x / (1.0f + std::exp(-t));
    }
    return output;
}
