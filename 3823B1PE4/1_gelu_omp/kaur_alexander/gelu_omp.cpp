#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

const float SQRT_2_OVER_PI = 0.7978845608f;
const float GELU_COEF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size); 

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float x = input[i];
pow
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);

        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}