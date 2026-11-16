#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "gelu_ocl.h"

static float GeluRef(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

int main(int argc, char** argv) {
    const int platform = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t n = 134217728;
    std::vector<float> input(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (auto& v : input) v = dist(gen);

    auto out = GeluOCL(input, platform);

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        out = GeluOCL(input, platform);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    double max_err = 0.0;
    for (size_t i = 0; i < n; i += 997) max_err = std::max(max_err, (double)std::fabs(out[i] - GeluRef(input[i])));
    std::printf("size=%zu time=%.6f s max_abs_err=%.3e %s\n", n, time, max_err, max_err < 1e-4 ? "OK" : "FAIL");
    return max_err < 1e-4 ? 0 : 1;
}
