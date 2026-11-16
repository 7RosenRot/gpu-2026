#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float scale = std::sqrt(2.0f / std::acos(-1.0f));

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(input.size()); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(scale * (x + 0.044715f * x * x * x)));
    }

    return output;
}
