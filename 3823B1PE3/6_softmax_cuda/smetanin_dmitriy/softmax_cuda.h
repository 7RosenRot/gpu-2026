#ifndef SMETANIN_DMITRIY_SOFTMAX_CUDA_H
#define SMETANIN_DMITRIY_SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif
