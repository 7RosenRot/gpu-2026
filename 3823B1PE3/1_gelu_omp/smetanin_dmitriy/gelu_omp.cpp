#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <limits>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float scale = 0.7978845608028654f;
    const auto size = static_cast<std::ptrdiff_t>(input.size());
    const float* values = input.data();
    std::vector<float> result(input.size());
    float* output = result.data();

    #pragma omp parallel for schedule(static) if(size >= 32768)
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        const float x = values[i];
        const float z = scale * x * (1.0f + 0.044715f * x * x);
        const float e = std::exp(-2.0f * std::abs(z));
        const float weight = (x < 0.0f ? e : 1.0f) / (1.0f + e);
        output[i] = x == -std::numeric_limits<float>::infinity() ? -0.0f : x * weight;
    }

    return result;
}
