#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "softmax_cuda.h"

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 4096;
    std::vector<float> input(size_t(rows) * cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : input) v = dist(gen);

    auto out = SoftmaxCUDA(input, rows);

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        out = SoftmaxCUDA(input, rows);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    double max_err = 0.0;
    for (int r = 0; r < rows; r += std::max(1, rows / 64)) {
        const float* x = &input[size_t(r) * cols];
        double m = *std::max_element(x, x + cols), s = 0.0;
        for (int j = 0; j < cols; ++j) s += std::exp(double(x[j]) - m);
        for (int j = 0; j < cols; ++j)
            max_err = std::max(max_err, std::fabs(std::exp(double(x[j]) - m) / s - out[size_t(r) * cols + j]));
    }
    std::printf("rows=%d cols=%d time=%.6f s max_abs_err=%.3e %s\n", rows, cols, time, max_err,
                max_err < 1e-5 ? "OK" : "FAIL");
    return max_err < 1e-5 ? 0 : 1;
}
