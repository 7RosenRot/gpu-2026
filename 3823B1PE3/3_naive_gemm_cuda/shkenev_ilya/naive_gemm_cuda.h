#ifndef SHKENEV_ILYA_NAIVE_GEMM_CUDA_H
#define SHKENEV_ILYA_NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif
