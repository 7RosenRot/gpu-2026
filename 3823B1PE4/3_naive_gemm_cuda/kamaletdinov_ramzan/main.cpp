#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "naive_gemm_cuda.h"

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 4096;
    std::vector<float> a(size_t(n) * n), b(size_t(n) * n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a) v = dist(gen);
    for (auto& v : b) v = dist(gen);

    auto c = NaiveGemmCUDA(a, b, n);

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        c = NaiveGemmCUDA(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    double max_err = 0.0;
    std::uniform_int_distribution<int> idx(0, n - 1);
    for (int t = 0; t < 256; ++t) {
        int i = idx(gen), j = idx(gen);
        double ref = 0.0;
        for (int k = 0; k < n; ++k) ref += double(a[size_t(i) * n + k]) * b[size_t(k) * n + j];
        max_err = std::max(max_err, std::fabs(ref - c[size_t(i) * n + j]));
    }
    const double tol = 1e-5 * n;
    std::printf("n=%d time=%.6f s max_abs_err=%.3e %s\n", n, time, max_err, max_err < tol ? "OK" : "FAIL");
    return max_err < tol ? 0 : 1;
}
