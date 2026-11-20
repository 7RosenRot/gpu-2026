#ifndef SMETANIN_DMITRIY_GEMM_CUBLAS_H
#define SMETANIN_DMITRIY_GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n);

#endif
