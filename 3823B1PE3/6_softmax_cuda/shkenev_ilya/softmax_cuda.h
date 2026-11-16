#ifndef SHKENEV_ILYA_SOFTMAX_CUDA_H
#define SHKENEV_ILYA_SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif
