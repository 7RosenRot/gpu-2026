#ifndef SMETANIN_DMITRIY_NAIVE_GEMM_CUDA_H
#define SMETANIN_DMITRIY_NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n);

#endif
