#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(input.size()); ++i) {
        const float x = input[static_cast<std::size_t>(i)];
        const float x3 = x * x * x;
        const float argument = kSqrtTwoOverPi * (x + kCubicCoefficient * x3);
        output[static_cast<std::size_t>(i)] = 0.5f * x * (1.0f + std::tanh(argument));
    }

    return output;
}
